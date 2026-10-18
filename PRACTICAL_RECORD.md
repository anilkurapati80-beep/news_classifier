# News Category Classification Using Naive Bayes — Practical Record

**Current status (verified 2026-10-16):** Practical 1 training, model loading, Flask routes, and pytest succeeded locally. Flask loads the fitted pipeline once during app creation. `MLOPS.pdf` is not present in the project folder, so its exact steps cannot be cross-checked. Docker is not installed/on PATH; Docker build/run and both cloud deployments remain pending.

## Practical 1 — Converting an ML Model into a Flask Application

**Aim:** Train a TF-IDF and Multinomial Naive Bayes pipeline and expose it through Flask web and REST interfaces.

**Requirements:** Python, dependencies in `requirements.txt`, and access to Hugging Face AG News (or another documented real labeled news dataset). The AG News page currently marks its license as unknown; check the dataset terms before redistribution.

**Procedure:** Create/use the project-local virtual environment; install requirements; when no local dataset exists, run `venv\Scripts\python.exe train_model.py --source ag-news` once; for later runs use the saved CSV with `--source csv`; verify label mapping and category distribution; review held-out evaluation; run Flask; inspect the webpage, health route, prediction API; run pytest.

**Important PowerShell commands:**

```powershell
cd .\news-category-classification
python -m venv venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
venv\Scripts\python.exe train_model.py --source csv --data data\news.csv --text-column text --label-column category --dataset-name https://huggingface.co/datasets/fancyzhx/ag_news
venv\Scripts\python.exe -m pytest -q
venv\Scripts\python.exe app.py
```

**Result/status:** Python 3.13.14 was used in project-local `venv`; verified versions: Flask 3.1.2, scikit-learn 1.7.2, pandas 2.3.2, NumPy 2.2.6, joblib 1.5.2, pytest 8.4.2, datasets 5.1.0. Cached Hugging Face feature metadata confirms label IDs `0=World, 1=Sports, 2=Business, 3=Sci/Tech`. Comparing the 120,000-row cached official training Parquet to `data/news.csv` confirmed 21 exact duplicate pairs removed; 119,979 records remain with no blank/missing rows. Counts: Sports 30,000; World 29,999; Business 29,997; Sci/Tech 29,983. A stratified 80/20 split with seed 42 produced 95,983 train and 23,996 test rows. Accuracy 0.912652; weighted precision 0.912298; weighted recall 0.912652; weighted F1 0.912386. `model.pkl` was regenerated and loaded; real sample output was Sports (estimated probability 0.9882). Pytest: 11 passed, including the startup-only model load check. Live Flask HTTP audit: GET / 200, GET /health 200 (`model_loaded: true`), POST /prediction 200, POST /api/predict 200. Empty, missing, non-string, malformed JSON, and oversized API text each returned 400. The real prediction was Sports; probabilities were mapped to the four class labels and summed to approximately 1. The live server log contained only expected access records, with no exception traceback. The Hugging Face loader attempt emitted an unauthenticated Hub warning and did not finish; training and evaluation were completed from the already downloaded CSV and no repeated data download was needed.

**Conclusion:** The local AG News + TF-IDF + MultinomialNB pipeline has been trained and served through the existing Flask app. Record the verified scores as one fixed-split result; the dataset page marks its license unknown, and Docker/AWS/GCP remain future practicals.

**Suggested screenshots:** Terminal showing category counts and actual metrics; browser homepage; successful prediction; `/health` JSON; passing pytest output. Capture manually; screenshots are not included.

## Practical 2 — Dockerizing the ML System

**Aim:** Package the Flask service and trained pipeline into a reproducible container.

**Requirements:** Docker Desktop/Engine and the generated `model.pkl` (present locally).

**Procedure/commands:** From the project folder, run `docker --version`, `docker info`, `docker build -t news-category-app:1.0 .`, then `docker run -d --name news-category-app --restart unless-stopped -p 5000:5000 news-category-app:1.0`. Inspect with `docker ps` and `docker logs news-category-app`; test `/`, `/health`, and `/api/predict`; stop/restart/remove as shown in README.

**Result/status:** Docker configuration is present and `model.pkl` exists. `docker --version` failed because `docker` was not recognized as a command. No image was built and no container was started; install/start Docker Desktop, reopen PowerShell, then run the README commands.

**Conclusion:** Dockerfile uses Gunicorn and expects the trained `model.pkl` at build time.

**Suggested screenshots:** Successful image build, container status/logs, browser/API health and prediction. Capture only after a real run.

## Practical 3 — AWS EC2 Deployment

**Aim:** Run the container on an Ubuntu EC2 instance.

**Requirements:** AWS account, selected region, SSH key, trusted client IP, and explicit cost approval before creating billable resources.

**Procedure:** Launch an Ubuntu VM in the chosen region; restrict SSH port 22 to your IP and port 5000 to intended clients; securely transfer the project; install Docker; build and run the image with port mapping and restart policy; inspect logs and verify remote routes. The detailed commands and placeholders are in README.md.

**Result/status:** Instructions prepared; no AWS account access, instance, or deployment was used. No IP or success is claimed.

**Conclusion:** Deployment remains a manual, billable step requiring the user's account and approval.

**Suggested screenshots:** Security group with restricted sources, instance status, SSH terminal/container logs, successful health and prediction response. Never include private key contents or credentials.

## Practical 4 — GCP Compute Engine Deployment

**Aim:** Run the same container on an Ubuntu Compute Engine VM.

**Requirements:** GCP project, billing account, Compute Engine API, trusted source CIDR, and explicit cost approval before creating billable resources.

**Procedure:** Select/create project and enable API; launch Ubuntu VM; configure a narrowly scoped TCP 5000 firewall rule; connect with browser SSH; install Docker; transfer/build and run the image; verify routes using the VM external address. Detailed local and VM commands are in README.md.

**Result/status:** Instructions prepared; no GCP project, VM, or deployment was used. No external IP or success is claimed.

**Conclusion:** Deployment remains pending required account setup and user approval of billable resources.

**Suggested screenshots:** Project/API configuration, restricted firewall, VM status, container logs, and successful health/prediction response. Redact identifying or sensitive account details.
