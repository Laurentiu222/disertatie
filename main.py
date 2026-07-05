from fastapi import FastAPI, Query, Body
import requests
import json
import re
import anthropic
from sentence_transformers import SentenceTransformer, util
from datetime import date
import os

app = FastAPI(title="Student Trainer API")

ADZUNA_APP_ID  = os.environ.get("ADZUNA_APP_ID",  "")
ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY", "")
JOOBLE_API_KEY = os.environ.get("JOOBLE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

embedder      = SentenceTransformer("all-MiniLM-L6-v2")
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# This service is stateless and never touches the database directly —
# InfinityFree (where the DB lives) blocks inbound requests from external
# services behind an anti-bot firewall, so PHP always initiates: it reads
# whatever data an endpoint here needs, sends it in the request body, and
# writes whatever comes back to the database itself.


# ── 1. Fetch jobs from Adzuna / Jooble ────────────────────────────────────────

@app.post("/fetch-jobs")
def fetch_jobs(
    query:    str = Query("software developer"),
    location: str = Query(""),
    pages:    int = Query(1),
    country:  str = Query("gb")
):
    ADZUNA_SUPPORTED = {"gb","us","au","de","fr","nl","pl","it","ca","za","sg","in","br","nz"}

    if country not in ADZUNA_SUPPORTED:
        jobs = _fetch_jooble(query, country, pages)
        source = "Jooble"
    else:
        jobs = []
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
                jobs.append({
                    "external_id":  str(j["id"]),
                    "title":        j.get("title", "")[:200],
                    "company":      j.get("company", {}).get("display_name", "")[:200],
                    "location":     j.get("location", {}).get("display_name", "")[:200],
                    "description":  j.get("description", ""),
                    "redirect_url": j.get("redirect_url", "")[:500],
                    "salary_min":   j.get("salary_min"),
                    "salary_max":   j.get("salary_max"),
                    "date_fetched": date.today().isoformat(),
                    "country":      country,
                })
        source = "Adzuna"

    return {"jobs": jobs, "source": source}


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

def _fetch_jooble(query: str, country: str, pages: int) -> list:
    country_name = JOOBLE_COUNTRY_NAMES.get(country, country.upper())
    jobs = []
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
            jobs.append({
                "external_id":  "jooble_" + str(j.get("id", "")),
                "title":        j.get("title", "")[:200],
                "company":      j.get("company", "")[:200],
                "location":     j.get("location", "")[:200],
                "description":  j.get("snippet", ""),
                "redirect_url": j.get("link", "")[:500],
                "salary_min":   None,
                "salary_max":   None,
                "date_fetched": date.today().isoformat(),
                "country":      country,
            })
    return jobs


# ── 2. Extract skills from a job description using Claude ────────────────────

@app.post("/extract-skills")
def extract_skills(payload: dict = Body(...)):
    description = payload.get("description", "")

    msg = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                "Extract all technical and professional skills from this job description. "
                "Return ONLY a valid JSON array of skill name strings, nothing else.\n"
                "Example: [\"Python\", \"React\", \"SQL\", \"Communication\"]\n\n"
                f"Job description:\n{description[:3000]}"
            )
        }]
    )
    raw = msg.content[0].text.strip()

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        extracted = json.loads(match.group()) if match else []

    skills = []
    for skill_name in extracted:
        skill_name = skill_name.strip()[:100]
        if not skill_name:
            continue
        skills.append({"name": skill_name, "category": _guess_category(skill_name)})

    return {"skills": skills}


# ── 3. Recommend courses for a student ────────────────────────────────────────

@app.post("/recommend")
def recommend(payload: dict = Body(...)):
    user_skills   = payload.get("user_skills", [])
    target_skills = payload.get("target_skills", [])
    playlists     = payload.get("playlists", [])
    top_n         = payload.get("top_n", 5)

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
