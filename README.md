# News Category Classification with Multinomial Naive Bayes

This Flask application classifies a headline or article using a fitted TF-IDF and Multinomial Naive Bayes pipeline. It supports the real AG News dataset from Hugging Face and its four labels: World, Sports, Business, and Sci/Tech. Training defaults to the existing local CSV so it does not repeatedly fetch the full dataset; use `--source ag-news` for the initial download.

## Practical manual and available data

`MLOPS.pdf` is not present in the project folder, so the practical steps cannot yet be checked against that file. AG News source, expected fields, label mapping, and license caveat are documented in [data/README.md](data/README.md). The evaluation values below are from an actual local training run, not illustrative examples.

## Practical 1 — Flask application

**Aim:** package text preprocessing and classification as one reusable pipeline and serve it through a web page and JSON API.

From PowerShell, enter the project folder and create a project-local Python environment (the current environment was verified with Python 3.13.14):

```powershell
cd .\news-category-classification
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The activation step is optional; commands can be run directly with `venv\Scripts\python.exe`. If PowerShell blocks activation, use that direct executable instead of changing system-wide policy.

If `data\news.csv` is not present yet, fetch AG News once with the official Hugging Face loader:

```powershell
venv\Scripts\python.exe train_model.py --source ag-news
```

For an existing downloaded CSV, train from that local copy so the full dataset is not fetched again:

```powershell
venv\Scripts\python.exe train_model.py --source csv --data data\news.csv --text-column text --label-column category --dataset-name https://huggingface.co/datasets/fancyzhx/ag_news
```

The loader obtains and validates the label names from Hugging Face metadata (`0=World`, `1=Sports`, `2=Business`, `3=Sci/Tech`), and uses only the official training split. It writes validated rows to `data\news.csv`; the cache is under project `.cache\`. The processed CSV is checked against the cached official Parquet data. Its 120,000 source rows contained 21 exact duplicate pairs, leaving 119,979 rows. The trainer reports the actual category counts, uses a reproducible stratified 80/20 split (seed 42), fits TF-IDF only on training rows, and saves the fitted pipeline and evaluation JSON after evaluation succeeds.

Verified local CSV training on 2026-10-16 used 119,979 records, split into 95,983 training rows and 23,996 held-out rows. Accuracy was 0.912652; weighted precision 0.912298, weighted recall 0.912652, and weighted F1 0.912386. Full per-class metrics and the confusion matrix are in `results\evaluation.json`. These scores describe that one fixed split and are not a guarantee of future predictions.

For a different real labeled CSV, pass `--source csv --data data\news.csv --text-column text --label-column category`. Change the split size or seed with `--test-size` and `--random-state`.

Run the development server locally:

```powershell
venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000`. At app startup, Flask loads the complete fitted `model.pkl` pipeline once; requests reuse it without training or refitting. The app listens on `0.0.0.0:5000` for container use. Routes: `GET /` web form, `POST /prediction` form submission, `GET /health` model status, and `POST /api/predict` JSON API. API input must be a JSON object with a nonempty string `text` up to 20,000 characters.

Example request (illustrative; actual category depends on the trained dataset):

```powershell
$body = @{ text = 'Central bank announces a change to interest rates' } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:5000/api/predict -Method Post -ContentType 'application/json' -Body $body
```

Illustrative response format (not a measured prediction):

```json
{
  "category": "<category learned from your CSV>",
  "probabilities": { "<category>": 0.72, "<another category>": 0.28 },
  "probability_note": "Model scores are estimates, not guaranteed certainty."
}
```

If the estimator lacks `predict_proba`, the probability fields are omitted. `400` means invalid input, `415` means wrong content type, `413` means oversized request, `503` means model unavailable, and `500` means prediction failed.

Run the automated unit suite from this directory:

```powershell
venv\Scripts\python.exe -m pytest -q
```

The tests use a tiny fake estimator only for route and validation behavior. Latest run: `11 passed`, including a check that startup loads the model once. Separately, the real `model.pkl` loaded and produced a `Sports` output for “The national team won the championship final after a close match.” The returned score for Sports was 0.9882; this is a model estimate, not a correctness guarantee.

Final Stage 2 live audit (2026-10-16): GET / returned 200; GET /health returned 200 with model_loaded: true; POST /prediction returned 200; POST /api/predict returned 200 with category Sports. The response probability keys were Business, Sci/Tech, Sports, and World; their values summed to 0.999999999999999 and were labelled as model estimates. Empty, missing, non-string, malformed JSON, and overlength API inputs each returned 400. The server log showed only these expected 200/400 request entries and no live exception traceback.

## Practical 2 — Docker

**Aim:** package the trained Flask service and model into a portable container. Build only after training generated `model.pkl`.

Check installation and daemon state:

```powershell
docker --version
docker info
```

Build and run:

```powershell
docker build -t news-category-app:1.0 .
docker run -d --name news-category-app --restart unless-stopped -p 5000:5000 news-category-app:1.0
docker ps
docker logs news-category-container
Invoke-RestMethod http://localhost:5000/health
$body = @{ text = 'A real headline from your selected test input' } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/api/predict -Method Post -ContentType 'application/json' -Body $body
```

Visit `http://localhost:5000/` for the page. Stop, restart, and remove the container with:

```powershell
docker stop news-category-app
docker start news-category-app
docker rm -f news-category-app
```

The image uses Gunicorn and Python 3.13 to match the verified training runtime. `.dockerignore` keeps local data, caches, secrets, tests, and evaluation files out of the build context; the trained `model.pkl` is included. Docker was not available during this verification (`docker` was not recognized as a command), so no image or container was built or run.

## Practical 3 — AWS EC2 (instructions only; not deployed)

**Aim:** run the same image on an Ubuntu EC2 virtual machine. No instance was created.

1. In AWS Console choose the region nearest users; Mumbai is `ap-south-1`. Select an Ubuntu LTS AMI and a small instance suitable for a classroom demonstration. Review current pricing, storage, data transfer, and free-tier eligibility before launch.
2. Create/download an SSH key pair and keep the private `.pem` file private. In the security group permit TCP 22 only from **My IP**. Permit TCP 5000 only from trusted client IP/CIDR ranges (or keep the service private and use an SSH tunnel). Do not use `0.0.0.0/0` for SSH.
3. Replace placeholders below. Commands marked **local PowerShell** run on your computer; **EC2 shell** runs after SSH login.

Local PowerShell — transfer project source (includes trained model; exclude secret keys):

```powershell
scp -i "<SSH_KEY_PATH.pem>" -r .\news-category-classification ubuntu@<EC2_PUBLIC_IP>:~/news-category-classification
ssh -i "<SSH_KEY_PATH.pem>" ubuntu@<EC2_PUBLIC_IP>
```

EC2 shell — install Docker on Ubuntu, build and launch:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo docker --version
cd ~/news-category-classification
sudo docker build -t news-category-app:1.0 .
sudo docker run -d --name news-category-container --restart unless-stopped -p 5000:5000 news-category-app:1.0
sudo docker ps
sudo docker logs news-category-container
curl -fsS http://localhost:5000/health
curl -fsS -X POST http://localhost:5000/api/predict -H 'Content-Type: application/json' -d '{"text":"A real headline"}'
```

From the local machine, verify `http://<EC2_PUBLIC_IP>:5000/`, `/health`, and the API using the PowerShell request example above with the public IP. To keep port 5000 private, use an SSH tunnel: `ssh -i "<SSH_KEY_PATH.pem>" -L 5000:localhost:5000 ubuntu@<EC2_PUBLIC_IP>`, then request `http://localhost:5000` locally. Stop/terminate unused instances and remove unused disks and public IP allocations to avoid charges.

## Practical 4 — Google Cloud Compute Engine (instructions only; not deployed)

**Aim:** run the same container on a Compute Engine VM. No GCP project or billable VM was created.

Create/select a project in Google Cloud Console, attach a billing account, and enable Compute Engine API. Billing is required and VM, disk, network egress, and static IP charges may apply. Review current prices; do not create the VM until you intentionally accept those charges. Create an Ubuntu LTS VM in a suitable region. Use a firewall rule scoped to trusted source CIDRs for TCP 5000; do not expose it to every source unless deliberately required. SSH through the browser or `gcloud compute ssh`.

Local PowerShell with Google Cloud CLI installed and authenticated — substitute your own values:

```powershell
$PROJECT_ID = '<GCP_PROJECT_ID>'
$ZONE = '<ZONE>'
$VM_NAME = '<VM_NAME>'
gcloud config set project $PROJECT_ID
gcloud compute scp --recurse .\news-category-classification "$VM_NAME`:~/news-category-classification" --zone $ZONE
gcloud compute ssh $VM_NAME --zone $ZONE
```

In the browser-based SSH window or remote VM shell:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo docker --version
cd ~/news-category-classification
sudo docker build -t news-category-app:1.0 .
sudo docker run -d --name news-category-container --restart unless-stopped -p 5000:5000 news-category-app:1.0
sudo docker ps
sudo docker logs news-category-container
curl -fsS http://localhost:5000/health
curl -fsS -X POST http://localhost:5000/api/predict -H 'Content-Type: application/json' -d '{"text":"A real headline"}'
```

Get the external IP in Console or with `gcloud compute instances describe <VM_NAME> --zone <ZONE> --format='get(networkInterfaces[0].accessConfigs[0].natIP)'`. Verify `http://<EXTERNAL_IP>:5000/`, `/health`, and `/api/predict` from a trusted client allowed by the firewall. Troubleshooting: a timeout usually indicates firewall or VM network configuration; check `sudo docker ps` and `sudo docker logs news-category-container` for container startup/model errors; ensure `model.pkl` was transferred and Python dependency versions match the training environment. Delete the VM and unneeded disks when done to stop ongoing charges.

## Lab record and viva summary

- **Practical 1:** Convert an ML workflow into a modular Flask app. Prepare a labeled CSV, fit TF-IDF and Naive Bayes as one pipeline, measure held-out metrics, then exercise the web form and API. The result is the trained pipeline plus an application; record actual output only after commands run.
- **Practical 2:** Package application dependencies and the trained model in Docker. Build, run, inspect health and logs, and manage the container lifecycle.
- **Practical 3:** Host the container on AWS EC2. Restrict inbound network rules, install Docker, transfer/build the image, and verify remotely. This guide does not imply deployment occurred.
- **Practical 4:** Repeat on GCP Compute Engine with project billing/API prerequisites and a narrowly scoped firewall rule. This guide does not imply deployment occurred.
- **Viva:** TF-IDF weights informative words; Multinomial Naive Bayes estimates a class from word counts/weights under a conditional independence assumption. A pipeline ensures the same vectorizer is used at inference. The test split estimates generalization and must remain separate from vectorizer fitting. Probability scores are not certainty.

## Files and artifact policy

`model.pkl` and the dataset/results are generated locally and ignored by Git. Keep a copy of the model artifact for deployment, or retrain from a documented dataset. Joblib artifacts should only be loaded from trusted sources and should be rebuilt when dependency versions change. The Docker image needs the generated `model.pkl` in this project directory at build time.
