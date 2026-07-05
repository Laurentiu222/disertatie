---
title: Disertatie Python Service
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

FastAPI microservice for the "disertatie" project (job fetching, Claude-based
skill extraction, course recommendations). Talks to the MySQL database through
`components/python_bridge.php` on the InfinityFree-hosted PHP site, since
InfinityFree does not allow remote MySQL connections.

Required Space secrets (Settings → Variables and secrets):
- `BRIDGE_URL` — e.g. https://yoursite.infinityfreeapp.com/components/python_bridge.php
- `BRIDGE_KEY` — must match `PYTHON_BRIDGE_KEY` in components/config.php
- `ANTHROPIC_API_KEY`
- `ADZUNA_APP_ID`
- `ADZUNA_APP_KEY`
- `JOOBLE_API_KEY`
