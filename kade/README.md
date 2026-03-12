# Kade Local Operator UI

Run locally:

```bash
python -m kade.ui.app
```

Or:

```bash
python -m kade.main --ui
```

This UI adds a unified chat + dashboard flow using deterministic runtime actions.
Natural-language prompts are interpreted into structured actions; deterministic engine payloads remain source-of-truth and are exposed via raw JSON panels.

API endpoints:

- `GET /api/dashboard`
- `POST /api/command`
- `POST /api/chat`
- `GET /api/history`
