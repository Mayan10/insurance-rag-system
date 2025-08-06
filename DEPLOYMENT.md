# Deployment Guide for Insurance RAG System

This guide will help you deploy the Insurance RAG System API for the hackathon submission.

## Quick Deploy Options

### 1. Deploy to Railway (Recommended)

1. **Fork/Clone the repository**
2. **Go to [Railway.app](https://railway.app)**
3. **Connect your GitHub repository**
4. **Add environment variables:**
   ```
   API_KEY=hackrx-2024-secret-key
   ```
5. **Deploy** - Railway will automatically detect the Python app

### 2. Deploy to Render

1. **Go to [Render.com](https://render.com)**
2. **Create a new Web Service**
3. **Connect your GitHub repository**
4. **Configure:**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host=0.0.0.0 --port=$PORT`
   - **Environment Variables:**
     ```
     API_KEY=hackrx-2024-secret-key
     ```

### 3. Deploy to Heroku

1. **Install Heroku CLI**
2. **Login to Heroku:**
   ```bash
   heroku login
   ```
3. **Create Heroku app:**
   ```bash
   heroku create your-app-name
   ```
4. **Set environment variables:**
   ```bash
   heroku config:set API_KEY=hackrx-2024-secret-key
   ```
5. **Deploy:**
   ```bash
   git push heroku main
   ```

### 4. Deploy to Vercel

1. **Go to [Vercel.com](https://vercel.com)**
2. **Import your GitHub repository**
3. **Configure as Python project**
4. **Add environment variables in Vercel dashboard**

## Local Testing

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variable
```bash
export API_KEY=hackrx-2024-secret-key
```

### 3. Run the Application
```bash
python app.py
```

### 4. Test the API
```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer hackrx-2024-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
      "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
    ]
  }'
```

## API Endpoints

### Main Endpoint
- **URL:** `POST /hackrx/run`
- **Alternative:** `POST /api/v1/hackrx/run`

### Health Check
- **URL:** `GET /health`
- **URL:** `GET /`

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | Authentication key | `hackrx-2024-secret-key` |
| `PORT` | Server port | `8000` |

## Troubleshooting

### Common Issues

1. **Memory Issues:**
   - Use `faiss-cpu` instead of `faiss-gpu`
   - Reduce batch sizes in the code

2. **Timeout Issues:**
   - Increase timeout limits in deployment platform
   - Optimize document processing

3. **Dependency Issues:**
   - Ensure all packages are in `requirements.txt`
   - Use Python 3.11+ for compatibility

### Performance Optimization

1. **For Large Documents:**
   - Increase memory allocation in deployment platform
   - Use batch processing for multiple questions

2. **For High Traffic:**
   - Enable caching
   - Use load balancing

## Security Notes

- Change the default API key in production
- Use environment variables for sensitive data
- Enable HTTPS in production
- Consider rate limiting for API endpoints

## Monitoring

The API includes logging for:
- Request processing
- Document downloads
- Question processing
- Error handling

Check your deployment platform's logs for debugging. 