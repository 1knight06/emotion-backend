from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import re

app = Flask(__name__)
CORS(app)

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

def clean_text(text):
    text = text.strip()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_prediction(text):
    if model is None:
        return None, "Model not loaded"
    text = clean_text(text)
    if len(text) < 3:
        return None, "Text too short"
    if len(text) > 2000:
        text = text[:2000]
    try:
        prediction   = model.predict([text])[0]
        proba        = model.predict_proba([text])[0]
        classes      = model.classes_
        confidence   = float(np.max(proba))
        all_emotions = {
            cls: round(float(p), 4)
            for cls, p in zip(classes, proba)
        }
        meta = EMOTION_META.get(
            prediction.lower(),
            {"emoji": "😶", "color": "#333333"}
        )
        return {
            "emotion": str(prediction),
            "confidence": round(confidence, 4),
            "emoji": meta["emoji"],
            "color": meta["color"],
            "all_emotions": all_emotions,
            "text_preview": text[:120] + ("..." if len(text) > 120 else ""),
        }, None
    except Exception as e:
        return None, str(e)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    result, error = run_prediction(data["text"])
    if error:
        return jsonify({"error": error}), 500
    return jsonify(result)

@app.route("/predict-file", methods=["POST"])
def predict_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename.endswith(".txt"):
        return jsonify({"error": "Only .txt files"}), 400
    try:
        content = file.read().decode("utf-8")
    except UnicodeDecodeError:
        content = file.read().decode("latin-1")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
    if not paragraphs:
        return jsonify({"error": "File appears empty"}), 400
    results = []
    for p in paragraphs[:20]:
        result, error = run_prediction(p)
        if result:
            results.append(result)
    return jsonify(results)

@app.route("/emotions/meta", methods=["GET"])
def emotion_meta():
    return jsonify(EMOTION_META)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)