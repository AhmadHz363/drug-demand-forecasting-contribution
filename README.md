# Hospital Drug Demand Intelligence Framework (HDDIF)

Decision-support software for **hospital pharmacy demand forecasting**. The system is the implementation accompanying the Master’s thesis *AI-Driven Drug Supply Chain for Hospital Departments* (Lebanese University, Faculty of Sciences).

HDDIF has two complementary pathways:

| Pathway | Name | Use when |
| --- | --- | --- |
| Warm start | **SHIELD-XR** — Structured Hybrid Interpretable Ensemble with eXtreme-event Reconciliation | The SKU already has usable demand history |
| Cold start | **CAMEO** — Cold-start Analog Metadata Embedding and Online updating | The SKU is newly listed and has little or no history |

A Next.js dashboard sits on top of a FastAPI backend and a PostgreSQL database. Pharmacists can ingest hospital Excel exports, inspect the drug and category registries, train both engines, and review weekly forecasts with accuracy diagnostics.

---

## Table of contents

1. [What you get](#what-you-get)
2. [Architecture](#architecture)
3. [Repository layout](#repository-layout)
4. [Prerequisites](#prerequisites)
5. [Quick start](#quick-start)
6. [Step-by-step setup](#step-by-step-setup)
7. [Run the project](#run-the-project)
8. [First-use workflow](#first-use-workflow)
9. [Hospital Excel format](#hospital-excel-format)
10. [Packages](#packages)
11. [Configuration](#configuration)
12. [API surface](#api-surface)
13. [Tests](#tests)
14. [Useful scripts](#useful-scripts)
15. [Troubleshooting](#troubleshooting)

---

## What you get

- **Data ingestion** — upload `.xlsx` / `.xls` / `.csv` hospital pharmacy exports. Raw lines are stored as-is; inpatient sales are cleaned into a daily demand panel.
- **Registries** — unique drugs and categories derived from receipts.
- **SHIELD-XR** — hospital-wide daily ensemble, weekly aggregation, hierarchical reconciliation, Syntetos–Boylan demand classes, spike handling, and hold-out accuracy.
- **CAMEO** — metadata-driven analog forecasting for new medicines, conformal intervals, and online updating as early observations arrive.
- **Auth** — JWT login/register, required for every dashboard module except `/health` and `/auth/*`.
- **API docs** — Swagger UI at `/docs` and ReDoc at `/redoc`.

Hospital pharmacy extracts used during the study are **not** in this repository (they contain operational patient-adjacent fields). You bring your own export that matches the column layout below.

---

## Architecture

```
┌─────────────────────┐         rewrite /api/backend/*          ┌────────────────────────┐
│  Next.js dashboard  │ ──────────────────────────────────────► │  FastAPI (uvicorn)     │
│  dashboard/         │         http://localhost:3000           │  backend/  :8000       │
│  React 19, Tailwind │                                         │  SQLAlchemy + Alembic  │
└─────────────────────┘                                         └───────────┬────────────┘
                                                                            │
                                                                            ▼
                                                                ┌────────────────────────┐
                                                                │  PostgreSQL            │
                                                                │  DrugForecastingDb     │
                                                                └────────────────────────┘
                                                                            │
                     ┌──────────────────────────────────────────────────────┤
                     ▼                                                      ▼
        SHIELD-XR artifacts                               CAMEO artifacts
        backend/src/app/forecasting/artifacts/            backend/src/app/cold_start/artifacts/
```

Training jobs (CAMEO metric net, SHIELD-XR LightGBM ensembles, optional TFT) run **inside the API process**. Keep the backend terminal open; large catalogs can take minutes to hours.

---

## Repository layout

```
.
├── README.md
├── backend/
│   ├── .env.example              # copy to .env
│   ├── requirements.txt          # Python packages
│   ├── alembic.ini               # migrations (run from backend/)
│   ├── main.py                   # uvicorn shim → app.main:app
│   ├── run.md                    # one-line API start reminder
│   ├── scripts/                  # seed, backfill, validation helpers
│   ├── tests/                    # pytest suite
│   └── src/app/
│       ├── main.py               # FastAPI application
│       ├── api/                  # HTTP routers
│       ├── core/                 # settings, DB session, JWT
│       ├── models/               # SQLAlchemy models
│       ├── migrations/           # Alembic revisions
│       ├── services/             # ingestion, enrichment, overview
│       ├── forecasting/          # SHIELD-XR
│       └── cold_start/           # CAMEO (+ legacy KNN/MAML modules)
└── dashboard/
    ├── .env.example              # copy to .env.local if needed
    ├── package.json
    ├── app/                      # Next.js App Router pages
    ├── components/               # UI by module
    └── lib/                      # API client, types, constants
```

The thesis LaTeX tree (`thesis-manuscript/`) is gitignored and is not required to run the software.

---

## Prerequisites

Install these **before** cloning is enough; you will use them in order below.

| Tool | Version | Why |
| --- | --- | --- |
| **Python** | **3.11** recommended (3.10–3.12). Avoid 3.13 until all ML wheels catch up (`learn2learn`, PyTorch extras). | API, models, migrations |
| **Node.js** | **20.9+** (Next.js 16) | Dashboard |
| **npm** | ships with Node | Frontend packages |
| **PostgreSQL** | 14+ | Application database |
| **Git** | any recent | Clone |

Optional:

- Apple Silicon: Xcode Command Line Tools (`xcode-select --install`) so LightGBM / scientific wheels build if a prebuilt wheel is missing.
- A GPU is **not** required. Training falls back to CPU. Set `TFT_ENABLE_MPS=true` only if you want PyTorch MPS on Mac.

Check versions:

```bash
python3 --version
node --version
npm --version
psql --version
```

---

## Quick start

If you already know the stack, this is the shortest path. Detailed explanations follow in [Step-by-step setup](#step-by-step-setup).

```bash
git clone <this-repo-url>
cd drug-demand-forecasting-contribution

# 1. Database
createdb DrugForecastingDb   # or use psql as shown below

# 2. Backend
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env         # edit DATABASE_URL / JWT_SECRET_KEY
export PYTHONPATH=src
alembic upgrade head
python scripts/create_user.py --email you@hospital.local --password 'ChooseALongPassword'

uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 3. Dashboard (second terminal)
cd dashboard
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), sign in, upload an Excel export, then train CAMEO and SHIELD-XR from the dashboard.

---

## Step-by-step setup

### 1. Clone the repository

```bash
git clone <this-repo-url>
cd drug-demand-forecasting-contribution
```

### 2. Install and start PostgreSQL

**macOS (Homebrew)**

```bash
brew install postgresql@16
brew services start postgresql@16
```

**Ubuntu / Debian**

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

Create the database and (if needed) the `postgres` role:

```bash
# macOS Homebrew often lets your OS user connect as postgres without a password
createdb DrugForecastingDb

# If createdb is not on PATH, use:
#   psql postgres -c 'CREATE DATABASE "DrugForecastingDb";'
```

If your `postgres` user has a password, put it in `DATABASE_URL` (see [Configuration](#configuration)).

Verify:

```bash
psql -d DrugForecastingDb -c '\conninfo'
```

### 3. Backend Python environment

Always work from `backend/` for API commands.

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Upgrade pip, then install every package listed in `requirements.txt`. The first install pulls PyTorch and can take several minutes.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `torch`, `lightning`, or `pytorch-forecasting` fail on your platform, install a CPU PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) first, then re-run `pip install -r requirements.txt`.

### 4. Backend environment file

```bash
cp .env.example .env
```

Edit `backend/.env`:

- Set `DATABASE_URL` to match your Postgres user, password, host, and database name.
- Replace `JWT_SECRET_KEY` with a long random string (even for local use if the machine is shared).

The settings class reads `backend/.env` automatically when the working directory is `backend/`.

### 5. Apply database migrations

Alembic must run from `backend/` so it finds `alembic.ini` and `src/app`.

```bash
cd backend
source .venv/bin/activate
export PYTHONPATH=src
alembic upgrade head
```

This creates tables for users, raw receipts, the enriched daily panel, drug/category registries, CAMEO drug attributes, forecast artifacts metadata, and overview KPIs.

Check current revision:

```bash
alembic current
```

You should see `20260905_0023` (or whatever is `head` after later migrations).

### 6. Create the first dashboard user

Registration is also available on the login page. Creating a user from the CLI is useful for a known admin account:

```bash
cd backend
source .venv/bin/activate
export PYTHONPATH=src
python scripts/create_user.py --email you@hospital.local --password 'ChooseALongPassword'
```

Password must be at least **8 characters**.

### 7. Dashboard (Node)

```bash
cd dashboard
npm install
```

The dashboard proxies `/api/backend/*` to `http://localhost:8000` by default. Only create `dashboard/.env.local` if the API lives elsewhere:

```bash
cp .env.example .env.local
# then set NEXT_PUBLIC_API_URL=http://127.0.0.1:8000  (or your host)
```

---

## Run the project

You need **two terminals**. Keep both running.

### Terminal A — API

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

`backend/main.py` is a shim that puts `src/` on `PYTHONPATH`, so `uvicorn main:app` is the supported command.

Confirm:

- Liveness: [http://localhost:8000/health](http://localhost:8000/health) → `{"status":"ok"}`
- Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Terminal B — dashboard

```bash
cd dashboard
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Unauthenticated visits redirect to `/login`.

| Script | Purpose |
| --- | --- |
| `npm run dev` | Local development (port 3000) |
| `npm run build` | Production bundle |
| `npm run start` | Serve the production bundle |
| `npm run lint` | ESLint |

---

## First-use workflow

Do these in order. Later modules depend on earlier ones.

### 1. Sign in

Use the account from `create_user.py`, or switch the login form to **register** (email + password ≥ 8 characters).

### 2. Upload hospital receipts — **Data Ingestion**

Go to **Data Ingestion** and upload the pharmacy Excel/CSV export.

The API:

1. Stores every row in `hospital_receipt_raw`.
2. Keeps inpatient patient-sale movements and builds `hospital_daily_demand_enriched` (daily demand plus hospital-activity features).

Large files can take several minutes. Watch the **backend terminal**; the dashboard proxies long uploads with a two-hour timeout and a 500 MB body limit.

### 3. Inspect registries — **Drug Registry** / **Category Registry**

Unique codes and categories appear after ingestion (and after the registry backfill if you run the helper script). Use **Overview** for system-wide counts.

If the registries look empty after a successful upload, backfill from raw receipts:

```bash
cd backend
source .venv/bin/activate
export PYTHONPATH=src
python scripts/backfill_registry_from_hospital_receipts.py
```

### 4. Seed CAMEO drug attributes (required for cold start)

CAMEO trains only on rows in `cameo_drugs` with `match_status = matched_source`, and each of those SKUs needs **at least 30 weeks** of enriched demand.

Load the pharmacological attribute workbook (sheets `drug_info_filled` and `match_report`):

```bash
cd backend
source .venv/bin/activate
export PYTHONPATH=src
python scripts/seed_drugs_from_excel.py --file /absolute/path/to/drugs_training_synthetic_filled_with_codes.xlsx
```

Add `--reset` to wipe `cameo_drugs` before insert.

Without this seed, **Cold Start → Train** will fail with “Need at least 5 matched_source drugs…”.

### 5. Train CAMEO — **Cold Start**

Open **Cold Start** and run training. The job fits the metric network, writes artifacts under `backend/src/app/cold_start/artifacts/`, and returns hold-out accuracy (WAPE / clipped accuracy, analog win rates, conformal coverage).

Then enter metadata for a **new** SKU (class, form, strength, route, …) to obtain an analog-based weekly launch forecast.

### 6. Train SHIELD-XR — **Forecasting**

Open **Forecasting** and start training. SHIELD-XR uses the **full enriched hospital panel** (not a single SKU). It compares the SHIELD-XR stack against plain Tweedie and L1 baselines, then stores artifacts under `backend/src/app/forecasting/artifacts/` (gitignored).

Retrain only when you ingest new history or change the engine; existing artifacts are reused unless you force a retrain from the UI.

### 7. Read forecasts and KPIs

- **Forecasting** — hospital-week total, reconciled per-drug breakdown, accuracy by Syntetos–Boylan class.
- **Overview** — ingestion volume, registry sizes, training status.
- **Cold Start** — per-drug analog accuracy table after CAMEO training.

---

## Hospital Excel format

Accepted extensions: `.xlsx`, `.xls`, `.csv`.

Headers are matched after stripping whitespace. Expected columns (hospital export names → stored fields):

| Excel header | Meaning |
| --- | --- |
| `DOC` | Document number |
| `LINE` | Line number |
| `CAT` | Category |
| `C.R` | Care / cost-center related code |
| `DATE` | Movement date (`DD/MM/YY`, Excel datetime, or ISO) |
| `MOV#` | Movement number |
| `Mov des` | Movement description |
| `CODE` | Drug / SKU code |
| `ARTICLE` | Article name |
| `M` | Unit / packing flag |
| `C.S` | Secondary classification |
| `QTY` | Quantity |
| `U.P` | Unit price |
| `T.P` | Total price |
| `MRN` | Patient identifier (stored; used only for activity counts) |
| `AD DATE` | Admission date |
| `R` / `U` | Auxiliary flags |
| `AGE` | Patient age |
| `DR` | Physician code |

The enrichment step keeps **inpatient patient-sale** movements and aggregates them to one row per `(drug_code, demand_date)`. Dates such as `00/00/00` are treated as missing.

---

## Packages

Pinned or bounded versions live in the lockfiles. Install from those files rather than picking packages ad hoc.

### Backend (`backend/requirements.txt`)

**Core API and database**

| Package | Role |
| --- | --- |
| `fastapi` | HTTP API |
| `uvicorn[standard]` | ASGI server |
| `pydantic` / `pydantic-settings` | Request models and `.env` config |
| `python-multipart` | File uploads |
| `sqlalchemy` | ORM |
| `alembic` | Schema migrations |
| `psycopg[binary]` | PostgreSQL driver (v3) |

**Auth**

| Package | Role |
| --- | --- |
| `bcrypt` | Password hashes |
| `python-jose[cryptography]` | JWT |
| `email-validator` | Email fields |

**Ingestion and numerics**

| Package | Role |
| --- | --- |
| `openpyxl` | Excel |
| `pandas` | Panels and cleaning |
| `numpy` | Arrays |

**Forecasting / ML**

| Package | Role |
| --- | --- |
| `scipy` / `statsmodels` | Statistical models (SARIMA and related) |
| `scikit-learn` | Scaling, metrics, helpers |
| `lightgbm` | Gradient boosting ensembles in SHIELD-XR |
| `shap` | Feature attributions |
| `mapie` | Conformal / interval helpers |
| `lifelines` | Survival-style utilities |
| `holidays` | Calendar features |
| `torch` | Deep learning runtime |
| `lightning` | Training loops |
| `pytorch-forecasting` | Temporal Fusion Transformer |
| `learn2learn` | Meta-learning (MAML path) |

**Tests**

| Package | Role |
| --- | --- |
| `pytest` | Test runner |
| `httpx` | API client for tests |

### Dashboard (`dashboard/package.json`)

**Runtime**

| Package | Role |
| --- | --- |
| `next` `16.2.9` | App Router, rewrites, production server |
| `react` / `react-dom` `19.2.4` | UI |
| `lucide-react` | Icons |
| `recharts` | Charts |

**Development**

| Package | Role |
| --- | --- |
| `typescript` | Types |
| `tailwindcss` `4` + `@tailwindcss/postcss` | Styling |
| `eslint` + `eslint-config-next` | Lint |
| `@types/node`, `@types/react`, `@types/react-dom` | Type packages |

Install with `npm install` so `package-lock.json` is respected.

---

## Configuration

### Backend environment variables

Loaded from `backend/.env` (see `backend/src/app/core/config.py`).

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+psycopg://postgres@localhost:5432/DrugForecastingDb` | SQLAlchemy URL (**must** use `+psycopg`, not `psycopg2`) |
| `APP_NAME` | `Drug Receipts API` | OpenAPI title |
| `JWT_SECRET_KEY` | placeholder | Sign access tokens |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `1440` (24 h) | Token lifetime |
| `AUTH_ENABLED` | `true` | When `false`, protected routers skip JWT (tests set this) |
| `TFT_ENABLE_MPS` | unset | `true` to allow Apple GPU for TFT |

URL with a password:

```text
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/DrugForecastingDb
```

### Dashboard environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Origin of FastAPI (used by Next rewrites and large multipart uploads) |

### Ports and CORS

| Service | Port |
| --- | --- |
| FastAPI | `8000` |
| Next.js | `3000` |

CORS allows `localhost`, `127.0.0.1`, and `[::1]` on any port for local development.

---

## API surface

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs).

Send `Authorization: Bearer <access_token>` on every route except `/health` and `/auth/*`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness |
| `POST` | `/auth/register` | Create user |
| `POST` | `/auth/login` | JWT |
| `GET` | `/auth/me` | Current user |
| `POST` | `/upload-hospital-receipts` | Excel/CSV ingest |
| `GET` | `/overview` | KPI snapshot |
| `GET` | `/drugs`, `/categories` | Registries |
| `GET` | `/cold-start/status` | CAMEO artifact status |
| `POST` | `/cold-start/train-cameo` | Train CAMEO (`train-embedder` / `train-maml` are aliases) |
| `POST` | `/cold-start/predict` | New-SKU forecast |
| `GET` | `/forecasting/status` | SHIELD-XR status |
| `POST` | `/forecasting/train` | Train SHIELD-XR |
| `POST` | `/forecasting/holdout` | Hold-out evaluation |
| `GET` | `/forecasting/performance` | Monitoring payload |

---

## Tests

From `backend/` with the venv active. `tests/conftest.py` sets `AUTH_ENABLED=false` so existing integration tests do not need a JWT.

```bash
cd backend
source .venv/bin/activate
export PYTHONPATH=src
pytest tests/ -v --tb=short
```

Run a subset:

```bash
pytest tests/test_auth.py tests/test_cameo_cold_start.py tests/test_shield_xr.py -v
```

Some tests expect a reachable Postgres (`DATABASE_URL`) and applied migrations. If a test fails on connection, confirm `alembic upgrade head` succeeded against the same URL.

---

## Useful scripts

Run all of these from `backend/` with `source .venv/bin/activate` and `export PYTHONPATH=src`.

| Script | What it does |
| --- | --- |
| `scripts/create_user.py` | Insert a bcrypt-hashed dashboard user |
| `scripts/seed_drugs_from_excel.py` | Load CAMEO `cameo_drugs` from the attribute workbook |
| `scripts/backfill_registry_from_hospital_receipts.py` | Rebuild `categories` / `drugs` / `drug_receipts` from raw ingest |
| `scripts/reingest_hospital_excel.py` | Re-run enrichment from a file on disk |
| `scripts/validate_forecast_accuracy.py` | Offline accuracy checks |
| `scripts/validate_forecasting_step8.py` | Forecasting pipeline sanity |
| `scripts/validate_cold_start_step1.py` / `validate_cold_start_step6.py` | Cold-start checkpoints |

---

## Troubleshooting

**`connection refused` / `password authentication failed` for Postgres**  
The default URL uses user `postgres` and no password. Create that role or change `DATABASE_URL`. On Homebrew, connecting as your macOS user (`postgresql+psycopg://YOUR_OS_USER@localhost:5432/DrugForecastingDb`) often works without a password.

**`alembic: command not found`**  
The venv is not active, or packages were installed with a different Python. `which alembic` should point at `backend/.venv/bin/alembic`.

**`ModuleNotFoundError: app`**  
Export `PYTHONPATH=src` from `backend/`, or start the API with `uvicorn main:app` so the shim runs.

**Dashboard login works but API calls 401**  
Backend `AUTH_ENABLED` is true (default) and the JWT is missing or expired. Sign out and sign in again. Confirm `NEXT_PUBLIC_API_URL` matches the running API.

**CORS errors**  
Use `http://localhost:3000` (or `127.0.0.1`) consistently. Mixed hostnames can look like a cross-origin request.

**Upload or training dies with a proxy/network error**  
The job may still be running in the API process. Check the uvicorn terminal. Refresh the page or retry once the log shows completion. Training and ingest are intentionally long-running.

**CAMEO training: “Need at least 5 matched_source drugs”**  
Run `seed_drugs_from_excel.py`, then confirm those codes exist in `hospital_daily_demand_enriched` with ≥ 30 weeks of history.

**SHIELD-XR training: “No enriched training data found”**  
Upload a hospital export on **Data Ingestion** first.

**`pip install` fails on `torch` / `learn2learn`**  
Use Python 3.11, not 3.13. Install CPU PyTorch from the official index, then retry `requirements.txt`.

**Port already in use**  
```bash
lsof -i :8000
lsof -i :3000
```
Stop the old process or pass `--port` / `-p` to uvicorn / Next.

**Reset the database (destructive)**  

```bash
dropdb DrugForecastingDb
createdb DrugForecastingDb
cd backend && source .venv/bin/activate && export PYTHONPATH=src && alembic upgrade head
```

Then recreate a user and re-ingest files. Training artifacts on disk are separate; delete `backend/src/app/cold_start/artifacts/*` and `backend/src/app/forecasting/artifacts/` if you also want a clean model slate (keep `.gitkeep` if present).

---

## License and data

This repository is research software for a university thesis. Hospital extracts and trained model weights are not published here. Do not commit `.env` files, Excel dumps, or artifact directories — they are gitignored.
