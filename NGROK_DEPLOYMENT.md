# 🚀 ngrok Deployment Guide (Local + Public URL)

## What is ngrok?

ngrok creates a secure tunnel to your localhost, making your local API accessible from the internet with a public HTTPS URL. This is perfect for hackathon submissions!

## Step-by-Step Setup

### 1. Install ngrok

**Option A: Download from ngrok.com**
1. Go to [ngrok.com](https://ngrok.com)
2. Sign up for a free account
3. Download ngrok for your OS
4. Extract and add to your PATH

**Option B: Using Homebrew (macOS)**
```bash
brew install ngrok
```

**Option C: Using pip**
```bash
pip install pyngrok
```

### 2. Authenticate ngrok

After signing up, get your authtoken from [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken)

```bash
ngrok config add-authtoken YOUR_AUTHTOKEN_HERE
```

### 3. Start Your Local API

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export API_KEY=hackrx-2024-secret-key

# Start the API
python app.py
```

Your API will be running at `http://localhost:8000`

### 4. Create ngrok Tunnel

In a new terminal window:

```bash
ngrok http 8000
```

You'll see output like:
```
Session Status                online
Account                       your-email@example.com
Version                       3.x.x
Region                        United States (us)
Latency                       51ms
Web Interface                 http://127.0.0.1:4040
Forwarding                    https://abc123.ngrok.io -> http://localhost:8000
```

**Your public URL is:** `https://abc123.ngrok.io`

### 5. Test Your ngrok URL

```bash
# Test health endpoint
curl https://abc123.ngrok.io/health

# Test main endpoint
curl -X POST "https://abc123.ngrok.io/hackrx/run" \
  -H "Authorization: Bearer hackrx-2024-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
      "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
    ]
  }'
```

## ngrok Free Tier Limits

✅ **What's Included:**
- 1 tunnel per session
- HTTPS support
- Custom subdomains (random)
- Web interface for monitoring
- Request inspection

⚠️ **Limitations:**
- URL changes each time you restart ngrok
- 40 connections per minute
- Session timeout after 2 hours of inactivity

## Keeping ngrok Running

### Option 1: Keep Terminal Open
- Keep the ngrok terminal window open
- Don't close it during hackathon evaluation

### Option 2: Use nohup (Linux/macOS)
```bash
nohup ngrok http 8000 > ngrok.log 2>&1 &
```

### Option 3: Use screen/tmux
```bash
# Start a new screen session
screen -S ngrok

# Run ngrok
ngrok http 8000

# Detach: Ctrl+A, then D
# Reattach: screen -r ngrok
```

## Monitoring

### ngrok Web Interface
Visit: `http://localhost:4040`
- View all requests
- Inspect request/response details
- Monitor traffic

### Check ngrok Status
```bash
# Check if ngrok is running
ps aux | grep ngrok

# View ngrok logs
tail -f ngrok.log
```

## Troubleshooting

### ngrok URL Not Working
1. Check if ngrok is running: `ps aux | grep ngrok`
2. Verify local API is running: `curl http://localhost:8000/health`
3. Check ngrok web interface: `http://localhost:4040`

### Connection Issues
1. Restart ngrok: `pkill ngrok && ngrok http 8000`
2. Check firewall settings
3. Verify port 8000 is not blocked

### URL Changed
- ngrok URLs change when you restart
- Update your submission URL if needed
- Consider using ngrok with custom domains (paid feature)

## Advantages of ngrok

✅ **Instant Deployment** - No waiting for cloud deployment
✅ **Full Control** - Your local environment
✅ **Easy Debugging** - Direct access to logs
✅ **No Cost** - Completely free
✅ **HTTPS Support** - Required for hackathon
✅ **Real-time Monitoring** - See all requests

## Disadvantages

⚠️ **URL Changes** - New URL each restart
⚠️ **Session Limits** - 2-hour timeout
⚠️ **Connection Limits** - 40/min on free tier
⚠️ **Local Dependency** - Your computer must stay on

## Final Submission URL

Your submission URL will be:
```
https://your-ngrok-url.ngrok.io/hackrx/run
```

Example: `https://abc123.ngrok.io/hackrx/run`

## Pre-Submission Checklist

- [ ] ngrok is running and stable
- [ ] Local API is running on port 8000
- [ ] Health endpoint works: `https://your-url.ngrok.io/health`
- [ ] Main endpoint works with test data
- [ ] HTTPS is working (ngrok provides this)
- [ ] Response time < 30 seconds
- [ ] Keep computer running during evaluation

## Backup Plan

If ngrok has issues during evaluation:
1. Have Render deployment ready as backup
2. Keep both local and cloud versions running
3. Provide both URLs in submission notes

## Security Note

ngrok URLs are public. For production, use proper authentication (which we have with API_KEY). 