---
title: Disertatie Python Service
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

Stateless FastAPI microservice for the "disertatie" project (job fetching,
Claude-based skill extraction, course recommendations). It never touches the
database directly — InfinityFree (where the DB and PHP site live) blocks
inbound requests from external services behind an anti-bot firewall. Instead,
the PHP site reads/writes the database itself and calls this service only for
computation, passing data in and getting results back in the same request.

Required Space secrets (Settings → Variables and secrets → **Secrets**, not
Variables — Variables are public on this Space):
- `ANTHROPIC_API_KEY`
- `ADZUNA_APP_ID`
- `ADZUNA_APP_KEY`
- `JOOBLE_API_KEY`
