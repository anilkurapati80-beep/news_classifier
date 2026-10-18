import app as app_module
from app import create_app


class FakeModel:
    classes_ = ["Business", "Sports"]

    def predict(self, texts):
        return ["Sports"]

    def predict_proba(self, texts):
        return [[0.15, 0.85]]


def client(model=FakeModel()):
    return create_app(model).test_client()


def test_health_with_model():
    response = client().get("/health")
    assert response.status_code == 200
    assert response.json["model_loaded"] is True


def test_model_is_loaded_once_at_startup(monkeypatch, tmp_path):
    loads = []

    def fake_load(path):
        loads.append(path)
        return FakeModel()

    monkeypatch.setattr(app_module, "MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(app_module.joblib, "load", fake_load)
    test_client = create_app().test_client()
    test_client.get("/health")
    test_client.post("/api/predict", json={"text": "A headline"})
    assert len(loads) == 1


def test_homepage():
    assert client().get("/").status_code == 200


def test_form_prediction():
    response = client().post("/prediction", data={"text": "The team won the final"})
    assert response.status_code == 200
    assert b"Predicted category: Sports" in response.data


def test_valid_api_prediction():
    response = client().post("/api/predict", json={"text": "The team won the final"})
    assert response.status_code == 200
    assert response.json["category"] == "Sports"
    assert response.json["probabilities"]["Sports"] == 0.85


def test_missing_empty_and_wrong_type():
    test_client = client()
    assert test_client.post("/api/predict", json={}).status_code == 400
    assert test_client.post("/api/predict", json={"text": "  "}).status_code == 400
    assert test_client.post("/api/predict", json={"text": 12}).status_code == 400


def test_text_is_trimmed_before_prediction():
    response = client().post("/api/predict", json={"text": "  The team won the final  "})
    assert response.status_code == 200
    assert response.json["category"] == "Sports"
    assert response.json["probabilities"]["Sports"] == 0.85


def test_non_json_request():
    assert client().post("/api/predict", data="text=hello").status_code == 415


def test_malformed_json():
    response = client().post("/api/predict", data="{not-json", content_type="application/json")
    assert response.status_code == 400
    assert "JSON object" in response.json["error"]


def test_input_too_long():
    response = client().post("/api/predict", json={"text": "x" * 20001})
    assert response.status_code == 400


def test_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "MODEL_PATH", tmp_path / "missing.pkl")
    test_client = create_app().test_client()
    assert test_client.get("/health").status_code == 503
    response = test_client.post("/api/predict", json={"text": "A headline"})
    assert response.status_code == 503
    assert response.json["error"] == "Model is unavailable. Train the model first."


def test_corrupted_model(monkeypatch, tmp_path):
    broken_model = tmp_path / "broken.pkl"
    broken_model.write_bytes(b"not a joblib model")
    monkeypatch.setattr(app_module, "MODEL_PATH", broken_model)
    test_client = create_app().test_client()
    assert test_client.get("/health").status_code == 503
