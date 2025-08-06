# 🚀 Render Deployment Guide (Free)

## Step-by-Step Deployment

### 1. Sign Up for Render
1. Go to [Render.com](https://render.com)
2. Click "Get Started for Free"
3. Sign up with your GitHub account
4. Verify your email

### 2. Create New Web Service
1. Click "New +" button
2. Select "Web Service"
3. Connect your GitHub repository: `https://github.com/Mayan10/insurance-rag-system`
4. Click "Connect"

### 3. Configure the Service
Fill in the following details:

**Basic Settings:**
- **Name:** `insurance-rag-system` (or any name you prefer)
- **Region:** Choose closest to you
- **Branch:** `main`
- **Root Directory:** Leave empty (default)

**Build & Deploy Settings:**
- **Runtime:** `Python 3`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn app:app --host=0.0.0.0 --port=$PORT`

**Environment Variables:**
Click "Advanced" and add:
```
API_KEY=hackrx-2024-secret-key
```

### 4. Deploy
1. Click "Create Web Service"
2. Wait for build to complete (5-10 minutes)
3. Your API will be live at: `https://your-app-name.onrender.com`

### 5. Test Your Deployment
```bash
curl -X POST "https://your-app-name.onrender.com/hackrx/run" \
  -H "Authorization: Bearer hackrx-2024-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
      "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
    ]
  }'
```

## Render Free Tier Limits

✅ **What's Included:**
- 750 hours/month of runtime
- 512 MB RAM
- Shared CPU
- Automatic HTTPS
- Custom domains
- Continuous deployment

⚠️ **Limitations:**
- Service sleeps after 15 minutes of inactivity
- First request after sleep may take 30-60 seconds
- 512 MB RAM limit (sufficient for our app)

## Troubleshooting

### Build Fails
1. Check the build logs in Render dashboard
2. Ensure all dependencies are in `requirements.txt`
3. Verify Python version compatibility

### Service Won't Start
1. Check the logs in Render dashboard
2. Verify the start command is correct
3. Ensure environment variables are set

### Memory Issues
1. The app is optimized for 512 MB RAM
2. If you encounter memory issues, contact me for optimization

### Slow Response Times
1. First request after sleep will be slow (30-60 seconds)
2. Subsequent requests will be fast
3. This is normal for free tier

## Monitoring

### View Logs
1. Go to your service in Render dashboard
2. Click "Logs" tab
3. Monitor for any errors

### Health Check
Visit: `https://your-app-name.onrender.com/health`

Expected response:
```json
{
  "status": "healthy",
  "rag_system_initialized": true,
  "timestamp": "2024-01-XX..."
}
```

## Cost
**Render Free Tier is completely free** for this use case. You won't be charged anything.

## Final URL for Submission
Once deployed, your submission URL will be:
```
https://your-app-name.onrender.com/hackrx/run
```

Replace `your-app-name` with the actual name you chose during deployment. 