# Chapter 16 online demonstration

The repository-level `render.yaml` deploys the saved workbench on Render's free web-service
plan. The saved mode needs no model key. The blueprint sets `CH16_ALLOW_LIVE=false` because the
teaching workbench has no public-user authentication. Keep that setting for a public demo. A
private deployment may set `CH16_ALLOW_LIVE=true` and add `ANTHROPIC_API_KEY` as a secret after
authentication, rate limits, and spending alerts are configured.

The free service is a teaching demonstration. Its local SQLite case store and LangGraph
checkpoint files are ephemeral. A restart, redeploy, or idle spin-down removes runs created in
the browser. Keep the committed saved traces for the no-key demonstration. Use a durable
database and task worker for retained cases.

After deployment, verify these endpoints:

```text
GET /
GET /health
GET /api/saved/first
GET /api/saved/later
```

The health response must report the analytics database and both saved traces. Do not place an
API key in the repository, image, build log, or browser code.
