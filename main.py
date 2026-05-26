from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import re

app = Flask(__name__)
CORS(app, origins="*", supports_credentials=False)

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

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = clean_text(str(data["text"]))
        if len(text) < 3:
            return jsonify({"error": "Text too short"}), 400
        if len(text) > 2000:
            text = text[:2000]

        if model is None:
            return jsonify({"error": "Model not loaded"}), 503

        prediction  = model.predict([text])[0]
        proba       = model.predict_proba([text])[0]
        classes     = model.classes_
        confidence  = float(np.max(proba))
        all_emotions = {
            str(cls): round(float(p), 4)
            for cls, p in zip(classes, proba)
        }
        meta = EMOTION_META.get(str(prediction).lower(), {"emoji": "😶", "color": "#333333"})

        result = {
            "emotion": str(prediction),
            "confidence": round(confidence, 4),
            "emoji": str(meta["emoji"]),
            "color": str(meta["color"]),
            "all_emotions": all_emotions,
            "text_preview": text[:120] + ("..." if len(text) > 120 else ""),
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict-file", methods=["POST", "OPTIONS"])
def predict_file():
    if request.method == "OPTIONS":
        return jsonify({}), 200
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
        try:
            text = clean_text(p)
            if len(text) < 3:
                continue
            prediction  = model.predict([text])[0]
            proba       = model.predict_proba([text])[0]
            classes     = model.classes_
            confidence  = float(np.max(proba))
            all_emotions = {str(cls): round(float(p2), 4) for cls, p2 in zip(classes, proba)}
            meta = EMOTION_META.get(str(prediction).lower(), {"emoji": "😶", "color": "#333333"})
            results.append({
                "emotion": str(prediction),
                "confidence": round(confidence, 4),
                "emoji": str(meta["emoji"]),
                "color": str(meta["color"]),
                "all_emotions": all_emotions,
                "text_preview": text[:120],
            })
        except Exception:
            continue
    return jsonify(results)

@app.route("/emotions/meta", methods=["GET"])
def emotion_meta():
    return jsonify(EMOTION_META)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)