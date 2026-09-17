# Clay Entropy

Clay Entropy is split into two product directions:

- **Clay Ghost** (`/#ghost`) — the concrete GTM product. It receives public business signals, evaluates whether the moment deserves attention, and routes to `relic`, `digital`, or `wait`.
- **Clay Protocol** (`/#protocol`) — the future-facing trust layer for agent identity, permission, reputation, and human handoff.

The root page (`/`) is intentionally minimal: two large choices, no dashboard chrome.

## Run

```bash
python3 app.py 4180
open http://127.0.0.1:4180/
```

The app uses only the Python standard library and SQLite. Generated state is ignored:

- `clay_entropy.sqlite3`
- `exports/`

## Clay API integration

Clay API credentials are server-side only. Copy `.env.example` to `.env`, populate the key, and export the variables before starting the server:

```bash
set -a
source .env
set +a
python3 app.py 4180
```

The current integration exposes:

- `GET /api/clay/status` — whether the server has a Clay key configured.
- `GET /api/clay/me` — authenticated Clay user/workspace response.
- `POST /api/clay/search` — creates a Clay advanced search from `{ "query": "..." }`.
- `POST /api/clay/routines/run` — starts a Clay routine from `{ "routine_id": "function:t_...", "items": [...] }`.
- `POST /hooks/clay` — verifies `X-Clay-Signature` using `CLAY_WEBHOOK_SIGNING_SECRET`.

Clay API keys use the `clay-api-key` header and never leave the server. See the current Clay developer docs: https://developers.clay.com/quickstart

## Checks

```bash
python3 check.py
```

The check exercises the signal engine, review queue, export package, STL output, and HTTP API. It uses temporary SQLite/export paths and does not require a Clay key.
