"""
FinBERT sentiment microservice.
Loads ProsusAI/finbert once at startup, serves HTTP on port 5001.
Model is cached in /root/.cache/huggingface (mounted volume).
"""
import logging
import os
import time

from flask import Flask, jsonify, request

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
_pipeline = None
_load_error = None


def _load_model():
    global _pipeline, _load_error
    try:
        logger.info("Loading ProsusAI/finbert — first run downloads ~400MB...")
        from transformers import pipeline as hf_pipeline
        _pipeline = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
        )
        logger.info("FinBERT loaded OK")
    except Exception as e:
        _load_error = str(e)
        logger.error(f"FinBERT load failed: {e}")


_LABEL_SCORE = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


@app.route("/health")
def health():
    if _pipeline:
        return jsonify({"ok": True, "model": "ProsusAI/finbert"})
    if _load_error:
        return jsonify({"ok": False, "error": _load_error}), 503
    return jsonify({"ok": False, "error": "model still loading"}), 503


@app.route("/sentiment", methods=["POST"])
def sentiment():
    if not _pipeline:
        return jsonify({"error": "model not ready", "score": 0.0, "label": "neutral"}), 503

    data = request.get_json(force=True)
    texts = [t.strip() for t in (data.get("texts") or []) if t.strip()]
    if not texts:
        return jsonify({"score": 0.0, "label": "neutral", "details": []})

    texts = texts[:10]  # cap at 10 headlines
    try:
        t0 = time.time()
        raw = _pipeline(texts, truncation=True, max_length=128)
        elapsed = round(time.time() - t0, 2)

        details = []
        weighted_scores = []
        for text, result in zip(texts, raw):
            # result is list of {label, score} for top_k=None
            best = max(result, key=lambda x: x["score"])
            label = best["label"].lower()
            conf = best["score"]
            signed = _LABEL_SCORE.get(label, 0.0) * conf
            weighted_scores.append(signed)
            details.append({"text": text[:120], "label": label, "score": round(conf, 3)})

        avg = sum(weighted_scores) / len(weighted_scores)
        label = "positive" if avg > 0.15 else "negative" if avg < -0.15 else "neutral"
        return jsonify({
            "score": round(avg, 4),
            "label": label,
            "details": details,
            "inference_ms": int(elapsed * 1000),
        })
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return jsonify({"error": str(e), "score": 0.0, "label": "neutral"}), 500


if __name__ == "__main__":
    _load_model()
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port)
