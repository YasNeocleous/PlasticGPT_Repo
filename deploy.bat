@echo off
REM PlasticGPT - Deploy to Google Cloud Run
REM Run this script from the project root folder

echo ================================================
echo PlasticGPT - Google Cloud Run Deployment
echo ================================================
echo.

REM Check if gcloud is installed
where gcloud >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: gcloud CLI not found!
    echo Install from: https://cloud.google.com/sdk/docs/install
    exit /b 1
)

REM Get project ID
for /f "tokens=*" %%i in ('gcloud config get-value project 2^>nul') do set PROJECT_ID=%%i
if "%PROJECT_ID%"=="" (
    echo ERROR: No Google Cloud project set.
    echo Run: gcloud config set project YOUR_PROJECT_ID
    exit /b 1
)

echo Using project: %PROJECT_ID%
echo.

REM Set variables
set SERVICE_NAME=plasticgpt
set REGION=us-central1

echo Deploying to Cloud Run...
echo This will take a few minutes on first deploy.
echo.

REM Deploy using source (Cloud Build will build the Docker image)
gcloud run deploy %SERVICE_NAME% ^
    --source . ^
    --region %REGION% ^
    --platform managed ^
    --allow-unauthenticated ^
    --memory 2Gi ^
    --cpu 1 ^
    --timeout 300 ^
    --min-instances 0 ^
    --max-instances 3 ^
    --set-secrets "GOOGLE_API_KEY=google-api-key:latest" ^
    --set-env-vars "GOOGLE_AI_MODEL=gemini-2.0-flash,GOOGLE_EMBED_MODEL=text-embedding-004,MAX_STARTUP_DOCS=0"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Deployment failed!
    echo.
    echo Make sure you have:
    echo 1. Created a secret called 'google-api-key' in Secret Manager
    echo 2. Enabled Cloud Run and Secret Manager APIs
    exit /b 1
)

echo.
echo ================================================
echo Deployment Complete!
echo ================================================
echo.
echo Your API is now live. Get the URL with:
echo   gcloud run services describe %SERVICE_NAME% --region %REGION% --format "value(status.url)"
echo.
