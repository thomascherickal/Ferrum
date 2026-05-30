# Serving the demo

Every dataset page fetches its model with a **relative** path
(`fetch('model.bin')`), so the `web/` directory **must be the web root**.
If you serve from the project root instead, you will get:

```
⚠ Failed to fetch model.bin: 404
```

## Correct commands

Run these from the **project root** — each one serves the `web/` folder as `/`:

```bash
# Python 3 (built in everywhere)
python3 -m http.server 8080 --directory web

# Node.js — note the directory argument
npx serve web
#   or
npx http-server web -p 8080

# Rust
cargo install basic-http-server      # one time
basic-http-server web
```

Then open **http://localhost:8080/** (not `/web/`).

## Wrong commands (these cause the 404)

```bash
python3 -m http.server 8080          # ✗ serves project root, model.bin is under web/
npx serve                            # ✗ same problem
basic-http-server .                  # ✗ same problem
```

When served from the project root the landing page still loads (it's at
`/web/index.html`), but the dataset pages resolve `model.bin` to
`/datasets/<slug>/model.bin`, which does not exist at the root — only under
`/web/`. Always make `web/` the served root.

## Quick check

After starting the server, this should return `200`:

```bash
curl -o /dev/null -w "%{http_code}\n" http://localhost:8080/datasets/iris/model.bin
```
