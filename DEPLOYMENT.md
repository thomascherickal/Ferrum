# 🚀 Ferrum Workspace Deployment Guide

This guide details how to deploy the interactive, WebAssembly-powered **Edge Generative AI Playgrounds** on static hosting services.

Because the playgounds run 100% locally inside the client's browser (compiled to static WASM and JS files), they can be hosted for **free** on any static host with zero backend server overhead.

---

## 1. Hosting Pre-requisites

Make sure you have compiled the latest WebAssembly package and copied the pre-trained `.bin` models into the web directory:

```bash
# 1. Compile WASM
bash scripts/build_wasm.sh

# 2. Copy pre-trained binaries
mkdir -p web/datasets/shell_oracle web/datasets/ambient_poet web/datasets/brand_alchemist
cp ../shell_oracle/shell_oracle.bin web/datasets/shell_oracle/model.bin
cp ../ambient_poet/ambient_poet.bin web/datasets/ambient_poet/model.bin
cp ../brand_alchemist/brand_alchemist.bin web/datasets/brand_alchemist/model.bin
```

---

## 2. Option A: GitHub Pages Deployment (Recommended)

You can publish the playground directly from your GitHub repository using GitHub Pages:

### Manual Setup
1. Commit the `web/` folder containing the compiled JS, WASM, and `.bin` dataset files to your master/main branch.
2. Go to your repository settings on GitHub -> **Pages**.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Choose the `master` (or `main`) branch and select the `/web` folder, then click **Save**.

Your playgrounds will be live in minutes at:
`https://<your-username>.github.io/Ferrum/`

---

## 3. Option B: Cloudflare Pages Deployment

Cloudflare Pages offers global CDN distribution and fast build pipelines:

1. Log into the Cloudflare Dashboard and select **Workers & Pages**.
2. Click **Create Application** -> **Pages** -> **Connect to Git**.
3. Select your repository.
4. Set the following build settings:
   - **Framework preset**: None
   - **Build command**: (Leave empty, or run `bash scripts/build_wasm.sh` if a Rust environment is present)
   - **Build output directory**: `web`
5. Click **Save and Deploy**.

---

## 4. Option C: Self-Hosted Static Server

Because the directory consists of purely static HTML, CSS, Javascript, WASM, and binary model assets, you can host it using Nginx, Apache, or simple node servers:

### Nginx Configuration Example
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    root /var/www/ferrum/web;
    index index.html;

    # Correct mime-type headers for WASM binaries
    location ~* \.wasm$ {
        add_header Content-Type application/wasm;
    }

    # Cache configurations for static binary assets
    location ~* \.(bin|wasm)$ {
        expires 1y;
        add_header Cache-Control "public, no-transform";
    }
}
```
