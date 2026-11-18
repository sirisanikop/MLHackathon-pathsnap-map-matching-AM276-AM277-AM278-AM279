# PathSnap — Map-Matching & Route Reconstruction

Noisy GPS trajectories → road-arc sequence. A learned candidate scorer (six models compared, LightGBM selected, Bayesian-tuned and then refined with more features and a ranking objective) feeds an HMM/Viterbi decoder over the directed road graph.

A FastAPI backend (laptop 1) serves predictions and computes accuracy against its private reference routes. A static evaluator UI (laptop 2) lists the evaluation trips, sends the selected trip ID and shows the predicted route and scores.

## Team

| Person | Name | SRN | GitHub |
|---|---|---|---|
| Person 1 | Spoorthi Poornachnadra | PES1UG24AM279 | Spoorthi-Poornachandra |
| Person 2 | Siri Krishna C R | PES1UG24AM276 | sirikrishna-dev |
| Person 3 | Soha Rafiq| PES1UG24AM278 | soharafiq |
| Person 4 | Siri Sanikope | PES1UG24AM277 | sirisanikop |

UE24CS352A Machine Learning hackathon, PES University.

## Headline results

| Stage | Train (CV / out-of-fold) top-1 | Student-test top-1 | Student-test macro-F1 | Continuity |
|---|---|---|---|---|
| Nearest-segment baseline | 0.553 | 0.545 | 0.366 | 0.805 |
| Round 1 scorer (LightGBM binary, 28 features) | 0.903 | 0.911 | 0.863 | 0.952 |
| Round 1 final (+ HMM/Viterbi + re-alignment) | 0.957 | 0.975 | 0.970 | 0.979 |
| Round 2 scorer (LightGBM LambdaRank, 46 features, K=24) | 0.918 | 0.932 | 0.908 | 0.960 |
| + HMM/Viterbi (dropout bridging) | 0.942 | 0.961 | 0.956 | 0.980 |
| + re-alignment to the predicted route (deployed) | 0.965 | 0.981 | 0.981 | 0.980 |

- Length-weighted route F1 of the deployed system: 0.993 (student-test), 0.981 (training out-of-fold).
- Trip-bootstrap 95% interval for the deployed top-1: [0.967, 0.992] on student-test.
- Labels are derived by aligning fixes to the supplied route and carry some noise (1.6% of moving fixes are labelled against the direction of travel). Under direction-aware labels the deployed top-1 is 0.965.
- Sources: `results/final_summary.json`, `results/realign_summary.json`, `results/validation_final.json`, `results/baseline_nearest.json`. Full analysis: `report/PathSnap_Technical_Report.pdf`.

## Repository layout

| Folder | Contents |
|---|---|
| `pathsnap/` | Library: data I/O and projection, label alignment (evaluation only), candidate generation, dataset building, model definitions, metrics, HMM/Viterbi, route-free inference pipeline |
| `scripts/` | EDA, dataset building, baseline, model training, Bayesian optimisation, HMM/decoder evaluation, validation, maps, smoke/robustness/preflight tests, report building |
| `api/` | FastAPI server (`server.py`) and the separate server-side evaluation step (`evaluation.py`) |
| `ui/` | Evaluator UI (`index.html`, `app.js`, Leaflet) |
| `models/` | Trained models (`*.joblib`) and `final_config.json` (model + HMM parameters used by the API) |
| `results/` | Every metric file: EDA tables, CV folds, per-model results, per-trip results, Optuna trial log, HMM grid/ablation |
| `figures/` | Plots used in the report |
| `maps/` | Interactive Folium maps (HTML) |
| `configs/` | Pipeline configuration summary |
| `report/`, `deck/` | Technical report PDF and pitch-deck sources |

## Installation (Windows)

```powershell
git clone https://github.com/Sonuaadi0706/pathsnap-map-matching.git
cd pathsnap-map-matching
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process Bypass` and activate again.

macOS / Linux: use `source .venv/bin/activate` instead of the activate line above.

## Data

The raw data is not committed. Place the organisers' dataset so that each trip has its own folder:

```
data/raw/
  train/<trip_id>/<trip_id>.track  (.nodes, .arcs, .route)
  student_test/<trip_id>/<trip_id>.track  (.nodes, .arcs, .route)
```

Point `PATHSNAP_DATA` at the folder that contains `train/` and `student_test/`:

```powershell
$env:PATHSNAP_DATA = "C:\path\to\pathsnap-map-matching\data\raw"
```

## Deployment (two laptops on the same network)

### Laptop 1 — API server

```powershell
.venv\Scripts\activate
$env:PATHSNAP_DATA = "C:\path\to\data\raw"
$env:PATHSNAP_PARTITION = "student_test"
python scripts/preflight.py
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

On the day of evaluation, set `PATHSNAP_PARTITION` to the folder name holding the evaluation trips.

Find the laptop's IP with `ipconfig` (IPv4 Address under the Wi-Fi adapter), and allow the port through the firewall (PowerShell as Administrator):

```powershell
netsh advfirewall firewall add rule name="PathSnap 8000" dir=in action=allow protocol=TCP localport=8000
```

Check from a second PowerShell window:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/trips
```

### Laptop 2 — evaluator UI

```powershell
curl.exe http://<LAPTOP1_IP>:8000/trips
cd ui
python -m http.server 8080
```

Open in the browser:

```
http://localhost:8080/?api=http://<LAPTOP1_IP>:8000
```

The UI lists the trips returned by `GET /trips`, sends `POST /predict {"trip_id": ...}`, draws the GPS fixes and predicted route on a map, and shows the server-side scores for three stages (classifier only, + HMM/Viterbi, + route re-alignment). The map tiles need internet access on laptop 2.

### LAN troubleshooting

- `curl.exe` works on laptop 1 but times out from laptop 2: the firewall is blocking Python. Allow Python on Private networks in Windows Defender Firewall.
- Both laptops must be on the same network. Some campus Wi-Fi networks block client-to-client traffic, so use a phone hotspot that both laptops join.
- Test `curl.exe http://<LAPTOP1_IP>:8000/trips` from laptop 2 before opening the UI.
- If laptop 1's IP changes (new network), run `ipconfig` again and update the URL.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | none | status, model name, partition |
| GET | `/trips` | none | trip IDs available for evaluation |
| POST | `/predict` | `{"trip_id": "00000009"}` | predicted route (arc IDs + line geometry), GPS fixes for display, timings, evaluation scores |

## Reproducing results

```powershell
python scripts/eda.py
python scripts/build_dataset.py
python scripts/baseline.py
foreach ($m in "logreg","gnb","knn","rf","lgbm","mlp") { python scripts/train_models.py $m }
python scripts/bayes_opt.py 40 2700
python scripts/train_models.py lgbm_tuned
python scripts/hmm_eval.py lgbm_tuned
python scripts/realign_eval.py
python scripts/validate_final.py
python scripts/smoke_test.py
```

The full round-2 pipeline (46-feature table, LambdaRank scorer, decoder tuning, report and deck building) uses the remaining scripts in `scripts/`. Seeds are fixed (42); folds are stored in `results/folds.json`.

## Leakage guarantees

- Prediction (`pathsnap/pipeline.py`) loads trips without the `.route` file and never imports `labels.py`.
- Evaluation (`api/evaluation.py`) runs after prediction and returns only aggregate scores. No reference geometry or labels reach the UI.
- Student-test routes are used only by evaluation code. Evaluator trips were never available during development.

## Metric definitions

- **Observation top-1 accuracy**: share of GPS fixes whose predicted arc equals the reference arc from monotone alignment of the fixes to the `.route` sequence. Fixes more than 60 m from the aligned route are excluded.
- **Macro-F1**: per-trip F1 averaged over the arcs in that trip, then averaged over trips.
- **Candidate miss rate**: share of scored fixes whose reference arc is not in the candidate set.
- **Route continuity**: share of consecutive fix pairs whose predicted arcs are identical or connected head to tail.
- **Route precision / recall / F1**: length-weighted overlap between predicted and reference arc sets.
