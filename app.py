"""Flask user interface and JSON API for the news category classifier."""
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(__import__("os").environ.get("MODEL_PATH", BASE_DIR / "model.pkl"))
MAX_TEXT_LENGTH = 20_000


def create_app(model=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_TEXT_LENGTH + 10_000
    if model is None:
        try:
            model = joblib.load(MODEL_PATH)
        except FileNotFoundError:
            app.logger.error("Model file is missing: %s", MODEL_PATH)
        except Exception as exc:
            app.logger.exception("Could not load model at %s", MODEL_PATH)
    app.config["CLASSIFIER"] = model

    def classifier():
        # The fitted pipeline is loaded once by create_app, never per request.
        return app.config["CLASSIFIER"]

    def predict(text):
        model_obj = classifier()
        if model_obj is None:
            return None, (jsonify(error="Model is unavailable. Train the model first."), 503)
        try:
            category = str(model_obj.predict([text])[0])
            result = {"category": category}
            if hasattr(model_obj, "predict_proba") and hasattr(model_obj, "classes_"):
                result["probabilities"] = {
                    str(label): float(probability)
                    for label, probability in zip(model_obj.classes_, model_obj.predict_proba([text])[0])
                }
                result["probability_note"] = "Model scores are estimates, not guaranteed certainty."
            return result, None
        except Exception:
            app.logger.exception("Prediction failed")
            return None, (jsonify(error="Prediction failed."), 500)

    @app.get("/")
    def home():
        return render_template("index.html", prediction=None, error=None, text="")

    @app.get("/health")
    def health():
        available = classifier() is not None
        return jsonify(status="ok" if available else "model_unavailable",
                       model_loaded=available), (200 if available else 503)

    @app.post("/prediction")
    def prediction_page():
        text = request.form.get("text", "")
        error = validate_text(text)
        if error:
            return render_template("index.html", prediction=None, error=error, text=text), 400
        normalized_text = text.strip()
        result, failure = predict(normalized_text)
        if failure:
            return render_template("index.html", prediction=None,
                                   error=failure[0].get_json()["error"], text=normalized_text), failure[1]
        return render_template("index.html", prediction=result, error=None, text=normalized_text)

    @app.post("/api/predict")
    def api_predict():
        if not request.is_json:
            return jsonify(error="Content-Type must be application/json."), 415
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Request body must be a JSON object."), 400
        text = payload.get("text")
        error = validate_text(text)
        if error:
            return jsonify(error=error), 400
        result, failure = predict(text.strip())
        if failure:
            return failure
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error=f"Request exceeds the {MAX_TEXT_LENGTH} character input limit."), 413

    return app


def validate_text(text):
    if not isinstance(text, str):
        return "Field 'text' is required and must be a string."
    normalized_text = text.strip()
    if not normalized_text:
        return "Field 'text' must not be empty."
    if len(normalized_text) > MAX_TEXT_LENGTH:
        return f"Text must be at most {MAX_TEXT_LENGTH} characters."
    return None


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
