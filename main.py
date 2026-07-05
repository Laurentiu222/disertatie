from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import requests
import json
import re
import random
import string
import anthropic
from sentence_transformers import SentenceTransformer, util
from datetime import date
import os

app = FastAPI(title="Student Trainer API")

ADZUNA_APP_ID  = os.environ.get("ADZUNA_APP_ID",  "")
ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY", "")
JOOBLE_API_KEY = os.environ.get("JOOBLE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# InfinityFree does not allow remote MySQL access, so all database work
# happens through components/python_bridge.php on the PHP site instead of
# a direct MySQL connection.
BRIDGE_URL = os.environ.get("BRIDGE_URL", "")   # e.g. https://yoursite.infinityfreeapp.com/components/python_bridge.php
BRIDGE_KEY = os.environ.get("BRIDGE_KEY", "")

embedder      = SentenceTransformer("all-MiniLM-L6-v2")
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def uid():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=20))


# InfinityFree's firewall silently drops requests without a browser-like
# User-Agent (e.g. the default "python-requests/x.x" one), so the bridge
# calls need to look like they come from a browser.
BRIDGE_HEADERS = {
    "X-Bridge-Key": BRIDGE_KEY,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
}


def bridge_get(action: str, params: dict = None) -> dict:
    resp = requests.get(
        BRIDGE_URL,
        params={"action": action, **(params or {})},
        headers=BRIDGE_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def bridge_post(action: str, json_body: dict = None) -> dict:
    resp = requests.post(
        BRIDGE_URL,
        params={"action": action},
        json=json_body or {},
        headers=BRIDGE_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# ── 1. Fetch jobs from Adzuna ─────────────────────────────────────────────────

@app.post("/fetch-jobs")
def fetch_jobs(
    query:    str = Query("software developer"),
    location: str = Query(""),
    pages:    int = Query(1),
    country:  str = Query("gb")
):
    stored = 0
    ADZUNA_SUPPORTED = {"gb","us","au","de","fr","nl","pl","it","ca","za","sg","in","br","nz"}

    if country not in ADZUNA_SUPPORTED:
        stored = _fetch_jooble(query, country, pages)
    else:
        for page in range(1, pages + 1):
            resp = requests.get(
                f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}",
                params={
                    "app_id":           ADZUNA_APP_ID,
                    "app_key":          ADZUNA_APP_KEY,
                    "results_per_page": 20,
                    "what":             query,
                    "where":            location,
                    "content-type":     "application/json",
                },
                timeout=15
            )
            resp.raise_for_status()

            for j in resp.json().get("results", []):
                result = bridge_post("upsert-job", {
                    "external_id":  str(j["id"]),
                    "title":        j.get("title", ""),
                    "company":      j.get("company", {}).get("display_name", ""),
                    "location":     j.get("location", {}).get("display_name", ""),
                    "description":  j.get("description", ""),
                    "redirect_url": j.get("redirect_url", ""),
                    "salary_min":   j.get("salary_min"),
                    "salary_max":   j.get("salary_max"),
                    "date_fetched": date.today().isoformat(),
                    "country":      country,
                })
                if result.get("inserted"):
                    stored += 1

    return {"stored": stored, "sources": ["Adzuna"] if country in ADZUNA_SUPPORTED else ["Jooble"]}


JOOBLE_COUNTRY_NAMES = {
    "ro": "Romania", "hu": "Hungary", "sk": "Slovakia", "hr": "Croatia",
}

@app.get("/debug-jooble")
def debug_jooble(query: str = Query("developer"), location: str = Query("Romania")):
    resp = requests.post(
        f"https://jooble.org/api/{JOOBLE_API_KEY}",
        json={"keywords": query, "location": location, "page": "1", "resultonpage": "5"},
        timeout=15
    )
    return {"status": resp.status_code, "body": resp.json()}

def _fetch_jooble(query: str, country: str, pages: int) -> int:
    country_name = JOOBLE_COUNTRY_NAMES.get(country, country.upper())
    stored = 0
    for page in range(1, pages + 1):
        resp = requests.post(
            f"https://jooble.org/api/{JOOBLE_API_KEY}",
            json={
                "keywords":     query,
                "location":     country_name,
                "page":         str(page),
                "resultonpage": "20",
            },
            timeout=15
        )
        resp.raise_for_status()

        for j in resp.json().get("jobs", []):
            result = bridge_post("upsert-job", {
                "external_id":  "jooble_" + str(j.get("id", uid())),
                "title":        j.get("title", ""),
                "company":      j.get("company", ""),
                "location":     j.get("location", ""),
                "description":  j.get("snippet", ""),
                "redirect_url": j.get("link", ""),
                "salary_min":   None,
                "salary_max":   None,
                "date_fetched": date.today().isoformat(),
                "country":      country,
            })
            if result.get("inserted"):
                stored += 1
    return stored


# ── 2. Extract skills from one job using Claude ───────────────────────────────

@app.post("/extract-skills/{job_id}")
def extract_skills(job_id: str):
    job_resp = bridge_get("job", {"id": job_id})
    job = job_resp.get("job")
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    msg = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                "Extract all technical and professional skills from this job description. "
                "Return ONLY a valid JSON array of skill name strings, nothing else.\n"
                "Example: [\"Python\", \"React\", \"SQL\", \"Communication\"]\n\n"
                f"Job description:\n{job['description'][:3000]}"
            )
        }]
    )
    raw = msg.content[0].text.strip()

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        extracted = json.loads(match.group()) if match else []

    skills_payload = []
    for skill_name in extracted:
        skill_name = skill_name.strip()[:100]
        if not skill_name:
            continue
        skills_payload.append({"name": skill_name, "category": _guess_category(skill_name)})

    tag_result = bridge_post("tag-job-skills", {"job_id": job_id, "skills": skills_payload})

    return {"job_id": job_id, "skills_extracted": tag_result.get("tagged", [])}


# ── 3. Batch-process all unprocessed jobs ────────────────────────────────────

@app.post("/extract-all-skills")
def extract_all_skills():
    jobs = bridge_get("unprocessed-jobs").get("job_ids", [])
    results = [extract_skills(jid) for jid in jobs]
    return {"processed": len(results)}


# ── 4. Recommend courses for a student ───────────────────────────────────────

@app.post("/recommend/{user_id}")
def recommend(
    user_id: str,
    job_id:  str = Query(None),
    top_n:   int = Query(5)
):
    user_skills = bridge_get("user-skills", {"user_id": user_id}).get("skills", [])

    job_skills_params = {"job_id": job_id} if job_id else {}
    target_skills = bridge_get("job-skills", job_skills_params).get("skills", [])

    if not target_skills:
        return {"recommendations": [], "message": "No target skills found."}

    # Compute skill gap via semantic similarity
    if user_skills:
        u_embs = embedder.encode(user_skills,   convert_to_tensor=True)
        t_embs = embedder.encode(target_skills, convert_to_tensor=True)
        sims   = util.cos_sim(t_embs, u_embs)          # (target × user)
        gap_skills = [
            target_skills[i]
            for i, s in enumerate(sims.max(dim=1).values.tolist())
            if s < 0.75
        ]
    else:
        gap_skills = target_skills

    if not gap_skills:
        return {"recommendations": [], "message": "You already match all required skills!"}

    playlists = bridge_get("tagged-playlists").get("playlists", [])

    if not playlists:
        return {"gap_skills": gap_skills, "recommendations": [], "message": "No tagged courses yet."}

    # Score each playlist by cosine similarity to the skill gap
    gap_emb = embedder.encode(", ".join(gap_skills), convert_to_tensor=True)
    scored  = []
    for p in playlists:
        p_emb = embedder.encode(p["skills"], convert_to_tensor=True)
        score = float(util.cos_sim(gap_emb, p_emb))
        scored.append({
            "playlist_id": p["id"],
            "title":       p["title"],
            "description": p["description"],
            "teaches":     p["skills"],
            "score":       round(score, 4)
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {
        "user_skills":   user_skills,
        "gap_skills":    gap_skills,
        "recommendations": scored[:top_n]
    }


# ── 5. Read-only helpers ──────────────────────────────────────────────────────

@app.get("/jobs")
def list_jobs(limit: int = Query(20), offset: int = Query(0)):
    return {"jobs": bridge_get("jobs", {"limit": limit, "offset": offset}).get("jobs", [])}


@app.get("/skills")
def list_skills():
    return {"skills": bridge_get("skills").get("skills", [])}


# ── Utility ───────────────────────────────────────────────────────────────────

def _guess_category(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["python","java","javascript","typescript","php","ruby","swift","kotlin","go","rust","sql","html","css","react","node","angular","vue","django","flask","spring","c++","c#"]):
        return "Programming"
    if any(k in n for k in ["machine learning","deep learning","tensorflow","pytorch","data science","pandas","numpy","scikit","nlp","ai","ml","llm","computer vision"]):
        return "Data & AI"
    if any(k in n for k in ["figma","photoshop","ux","ui","design","sketch","illustrator","wireframe"]):
        return "Design"
    if any(k in n for k in ["aws","azure","gcp","docker","kubernetes","devops","ci/cd","linux","cloud","terraform","jenkins"]):
        return "DevOps & Cloud"
    if any(k in n for k in ["management","leadership","communication","teamwork","agile","scrum","project","presentation"]):
        return "Soft Skills"
    return "Other"
