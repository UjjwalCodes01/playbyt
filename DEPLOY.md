# PlayByt — Google Cloud Run Deployment Guide

> **Frontend:** https://playbyt-i9ae.vercel.app  
> **Backend:** Google Cloud Run (this guide)

---

## Prerequisites

- Google account with billing enabled
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed
- Git repository cloned locally
- Node.js 18+ (for frontend rebuild)

---

## Part 1 — Google Cloud Project Setup

### 1.1 Install gcloud CLI (if not already installed)

```bash
# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# macOS
brew install --cask google-cloud-sdk

# Verify
gcloud --version
```

### 1.2 Login and create a project

```bash
# Login with your Google account
gcloud auth login

# Create a new dedicated project
gcloud projects create playbyt-backend --name="PlayByt"

# Set it as active
gcloud config set project playbyt-backend
```

### 1.3 Enable billing

> **Required** — Cloud Run won't work without billing. GCP gives **$300 free credits** for new accounts.

1. Open https://console.cloud.google.com/billing
2. Click **"Link a billing account"**
3. Select (or create) your billing account
4. Link it to the **playbyt-backend** project

### 1.4 Enable required APIs

```bash
gcloud services enable \
  cloudrun.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com
```

Expected output:
```
Operation "operations/acf.xxx" finished successfully.
```

---

## Part 2 — Deploy Backend to Cloud Run

### 2.1 Clone and enter the project

```bash
git clone https://github.com/something1703/PlayByt.git
cd PlayByt
```

### 2.2 Verify required files exist

```
PlayByt/
├── Dockerfile          ✅
├── start.sh            ✅
├── .dockerignore       ✅
├── main.py             ✅
├── server.py           ✅
├── sports_processor.py ✅
├── instructions.md     ✅
├── yolo11n-pose.pt     ✅ (6.2GB — required)
└── pyproject.toml      ✅
```

> **Note:** `yolo11n-pose.pt` must be in the root. If missing, download:
> ```bash
> wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
> ```

### 2.3 Deploy to Cloud Run

Replace the env var values below if your keys have changed:

```bash
gcloud run deploy playbyt-backend \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 1 \
  --set-env-vars="GEMINI_API_KEY=,STREAM_API_KEY=,STREAM_API_SECRET=
```

> - `--source .` — Cloud Build builds the Docker image from the local directory
> - `--memory 4Gi` — YOLO inference needs sufficient RAM
> - `--timeout 3600` — 1 hour timeout for long AI sessions
> - `--concurrency 1` — one active session per instance (AI agent is stateful)

### 2.4 Note the Service URL

Deployment output will show:
```
Service [playbyt-backend] revision [playbyt-backend-00001-abc] has been deployed
Service URL: https://playbyt-backend-xxxxxx.run.app
```

**Copy this URL** — you'll need it in Part 3.

### 2.5 Verify backend is running

```bash
curl https://playbyt-backend-xxxxxx.run.app/api/status
```

Expected:
```json
{"gemini": "idle", "yolo": "ok", "commentary_loop": "inactive", "frames_processed": 0}
```

---

## Part 3 — Connect Frontend to Backend

### 3.1 Update frontend production env

Edit `frontend/.env.production`:

```env
VITE_STREAM_API_KEY=m8ryv5sy48sa
VITE_API_URL=https://playbyt-backend-xxxxxx.run.app
```

Replace `xxxxxx` with your actual Cloud Run service URL.

### 3.2 Rebuild and redeploy frontend to Vercel

```bash
cd frontend
npm install
npm run build

# If Vercel CLI is installed:
vercel --prod

# Or push to GitHub — Vercel auto-deploys on push:
cd ..
git add frontend/.env.production
git commit -m "chore: update production API URL"
git push
```

### 3.3 Add Cloud Run URL to Vercel environment variables

1. Go to https://vercel.com/dashboard → **PlayByt project** → **Settings** → **Environment Variables**
2. Add:
   - Key: `VITE_API_URL`
   - Value: `https://playbyt-backend-xxxxxx.run.app`
   - Environment: **Production**
3. Click **Redeploy**

---

## Part 4 — Verify Full Stack

```bash
# 1. Open frontend
open https://playbyt-i9ae.vercel.app

# 2. Enter a name, select a role, click Create Room

# 3. Check backend logs in real time
gcloud run services logs read playbyt-backend \
  --region us-central1 \
  --limit 50
```

Expected logs:
```
🎮 PlayByt Agent initialized
🎙️ Commentary loop started
❓ Question loop started
```

---

## Part 5 — Update / Redeploy

Whenever you push code changes and want to redeploy:

```bash
cd PlayByt
git pull origin main
gcloud run deploy playbyt-backend \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 1 \
  --set-env-vars="GEMINI_API_KEY=AIzaSyCH6rn3x8mL4bdSH3NOvN2F7i3ctdp7tyU,STREAM_API_KEY=m8ryv5sy48sa,STREAM_API_SECRET=auxd6y2n3wknmxmjj8kx3gpd7febbjwxpm7abs2xwv7vvd24cvsqh3ucd4knjce8"
```

---

## Cost Estimate

| Resource | Free Tier | Paid |
|----------|-----------|------|
| Cloud Run CPU | 180,000 vCPU-seconds/month | $0.00001667/vCPU-sec |
| Cloud Run Memory | 360,000 GB-seconds/month | $0.000000185/GB-sec |
| Cloud Build | 120 min/day free | $0.003/build-min |
| Artifact Registry | 0.5 GB free | $0.10/GB/month |
| **Estimated monthly** | **$0 (light usage)** | **$5–15** |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Permission denied` on API enable | Attach billing account to project |
| `Build failed` — missing yolo11n-pose.pt | Add model file to repo root |
| `Container failed to start` | Check logs: `gcloud run services logs read playbyt-backend --region us-central1` |
| CORS error in browser | Confirm Cloud Run URL is in `server.py` `allow_origins` list |
| Frontend shows `localhost:8000` | Update `frontend/.env.production` VITE_API_URL and redeploy |
| Cold start timeout | Increase `--timeout` or set `--min-instances 1` (adds ~$15/month) |

---

## Environment Variables Reference

| Variable | Where | Value |
|----------|-------|-------|
| `GEMINI_API_KEY` | Cloud Run | Your Gemini API key |
| `STREAM_API_KEY` | Cloud Run + Vercel | `m8ryv5sy48sa` |
| `STREAM_API_SECRET` | Cloud Run only | Your Stream secret |
| `VITE_API_URL` | Vercel | `https://playbyt-backend-xxxxxx.run.app` |
| `VITE_STREAM_API_KEY` | Vercel | `m8ryv5sy48sa` |
