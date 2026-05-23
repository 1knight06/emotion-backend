from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get("MODEL_PATH", "text_emotion.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    model = None

EMOTION_META = {
    "anger":    {"emoji": "😠", "color": "#FF4B4B"},
    "disgust":  {"emoji": "🤮", "color": "#B8860B"},
    "fear":     {"emoji": "😨", "color": "#7C3AED"},
    "joy":      {"emoji": "😂", "color": "#F59E0B"},
    "neutral":  {"emoji": "😐", "color": "#6B7280"},
    "sadness":  {"emoji": "😔", "color": "#2563EB"},
    "shame":    {"emoji": "😳", "color": "#DC2626"},
    "surprise": {"emoji": "😮", "color": "#0891B2"},
}

class TextInput(BaseModel):
    text: str

class EmotionResult(BaseModel):
    emotion: str
    confidence: float
    emoji: str
    color: str
    all_emotions: dict
    text_preview: str

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_prediction(text: str) -> EmotionResult:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    text = clean_text(text)
    if len(text) < 3:
        raise HTTPException(status_code=400, detail="Text too short")
    if len(text) > 2000:
        text = text[:2000]
    try:
        prediction  = model.predict([text])[0]
        proba       = model.predict_proba([text])[0]
        classes     = model.classes_
        confidence  = float(np.max(proba))
        all_emotions = {
            cls: round(float(p), 4)
            for cls, p in zip(classes, proba)
        }
        meta = EMOTION_META.get(prediction.lower(), {"emoji": "😶", "color": "#333333"})
        return EmotionResult(
            emotion=prediction,
            confidence=round(confidence, 4),
            emoji=meta["emoji"],
            color=meta["color"],
            all_emotions=all_emotions,
            text_preview=text[:120] + ("..." if len(text) > 120 else ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=EmotionResult)
def predict(data: TextInput):
    return run_prediction(data.text)

@app.post("/predict-file", response_model=list[EmotionResult])
async def predict_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")
    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
    if not paragraphs:
        raise HTTPException(status_code=400, detail="File appears empty")
    paragraphs = paragraphs[:20]
    return [run_prediction(p) for p in paragraphs]

@app.get("/emotions/meta")
def emotion_meta():
    return EMOTION_META