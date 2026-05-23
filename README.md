<<<<<<< HEAD
# Emotion Detection API — Backend

FastAPI backend that serves your `text_emotion.pkl` model.

## Local Setup

```bash
cd backend
pip install -r requirements.txt

# Copy your model here
cp /path/to/text_emotion.pkl .

uvicorn main:app --reload --port 8000
```

Open http://localhost:8000/docs for interactive API docs.

## Deploy to Render (Free)

1. Push this `backend/` folder to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your repo
4. Set Build Command: `pip install -r requirements.txt`
5. Set Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Upload `text_emotion.pkl` via Render's Disk or store in repo
7. Deploy — you'll get a URL like `https://your-app.onrender.com`

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Check if API is alive |
| POST | /predict | Analyze a text string |
| POST | /predict-file | Upload a .txt file |
| GET | /emotions/meta | Get emoji + colors |

## Example Request

```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy today!"}'
```
=======
# emotion-backend
Emotion Detection API
>>>>>>> e153e9e27baf20e0bde4abfa71990aa08bef3128
