# PlasticGPT - Production Deployment Guide

## Quick Deploy to Google Cloud Run (5-10 minutes)

### Prerequisites
1. [Google Cloud account](https://console.cloud.google.com) with billing enabled
2. [gcloud CLI installed](https://cloud.google.com/sdk/docs/install)

---

### Step 1: Set Up Google Cloud (One-time)

Open PowerShell and run:

```powershell
# Login to Google Cloud
gcloud auth login

# Create a new project (or use existing)
gcloud projects create plasticgpt-prod --name="PlasticGPT"

# Set as current project
gcloud config set project plasticgpt-prod

# Enable required APIs
gcloud services enable run.googleapis.com secretmanager.googleapis.com cloudbuild.googleapis.com

# Enable billing (required - do this in the console)
# https://console.cloud.google.com/billing
```

---

### Step 2: Store Your API Key Securely

```powershell
# Create secret for your Google AI API key
# Replace YOUR_API_KEY with your actual key
echo "YOUR_GOOGLE_API_KEY" | gcloud secrets create google-api-key --data-file=-

# Grant Cloud Run access to the secret
gcloud secrets add-iam-policy-binding google-api-key `
    --member="serviceAccount:$(gcloud config get-value project)-compute@developer.gserviceaccount.com" `
    --role="roles/secretmanager.secretAccessor"
```

---

### Step 3: Deploy

```powershell
cd c:\Users\yason\Documents\GitHub\PlasticGPT_Repo

# Deploy to Cloud Run (takes 3-5 minutes first time)
gcloud run deploy plasticgpt `
    --source . `
    --region us-central1 `
    --allow-unauthenticated `
    --memory 2Gi `
    --timeout 300 `
    --set-secrets "GOOGLE_API_KEY=google-api-key:latest" `
    --set-env-vars "GOOGLE_AI_MODEL=gemini-1.5-flash,GOOGLE_EMBED_MODEL=text-embedding-004,MAX_STARTUP_DOCS=0"
```

---

### Step 4: Get Your URL

After deployment, you'll see something like:
```
Service URL: https://plasticgpt-xxxxx-uc.a.run.app
```

Test it:
```powershell
# Health check
curl https://plasticgpt-xxxxx-uc.a.run.app/health

# Test chat
curl -X POST https://plasticgpt-xxxxx-uc.a.run.app/api/chat `
    -H "Content-Type: application/json" `
    -d '{"question": "What is breast reconstruction?"}'
```

---

## Costs (Estimated)

| Service | Usage | Monthly Cost |
|---------|-------|-------------|
| Cloud Run | ~1000 requests | $0-5 |
| Gemini API | ~100K tokens | $0.50-2 |
| Secret Manager | 1 secret | $0.06 |
| **Total** | | **~$1-10/month** |

Cloud Run scales to zero when not in use = no cost when idle!

---

## Updating the App

After making changes, just redeploy:
```powershell
gcloud run deploy plasticgpt --source . --region us-central1
```

---

## Connecting Your Frontend

Update your frontend to use the Cloud Run URL:

```typescript
// In client/src/lib/api.ts or similar
const API_URL = "https://plasticgpt-xxxxx-uc.a.run.app";
```

---

## Troubleshooting

### View logs
```powershell
gcloud run logs read plasticgpt --region us-central1
```

### Check service status
```powershell
gcloud run services describe plasticgpt --region us-central1
```

### Redeploy with more memory (if needed)
```powershell
gcloud run deploy plasticgpt --source . --region us-central1 --memory 4Gi
```
