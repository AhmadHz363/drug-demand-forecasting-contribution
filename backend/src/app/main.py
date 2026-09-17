import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.categories import router as categories_router
from app.api.cold_start import router as cold_start_router
from app.api.deps import get_current_user
from app.api.drugs import router as drugs_router
from app.api.hospital_ingestion import router as hospital_ingestion_router
from app.api.overview import router as overview_router
from app.api.shield_xr_forecasting import router as shield_xr_forecasting_router
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title=settings.app_name,
    description=(
        "Drug demand forecasting API. "
        "Use **Swagger UI** at [`/docs`](/docs), **ReDoc** at [`/redoc`](/redoc), "
        "or fetch the schema at [`/openapi.json`](/openapi.json)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "auth", "description": "JWT login and user accounts."},
        {"name": "hospital-ingestion", "description": "Hospital Excel raw + enriched ingestion."},
        {"name": "Cold Start", "description": "CAMEO cold-start training and prediction."},
        {"name": "Forecasting", "description": "SHIELD-XR training and performance monitoring."},
        {"name": "drugs", "description": "Unique drugs from receipt ingestion."},
        {"name": "categories", "description": "Unique categories from receipt ingestion."},
        {"name": "overview", "description": "System-wide KPI snapshot."},
        {"name": "health", "description": "Service liveness."},
    ],
)

# Dashboard dev server may use localhost, 127.0.0.1, or [::1] depending on the browser/OS.
_LOCAL_DEV_ORIGIN_REGEX = r"http://(\[::1\]|localhost|127\.0\.0\.1)(:\d+)?"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://[::1]:3000",
    ],
    allow_origin_regex=_LOCAL_DEV_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_auth_required = [Depends(get_current_user)] if settings.auth_enabled else []

app.include_router(auth_router)
app.include_router(hospital_ingestion_router, dependencies=_auth_required)
app.include_router(cold_start_router, dependencies=_auth_required)
app.include_router(shield_xr_forecasting_router, dependencies=_auth_required)
app.include_router(drugs_router, dependencies=_auth_required)
app.include_router(categories_router, dependencies=_auth_required)
app.include_router(overview_router, dependencies=_auth_required)


@app.get("/health", tags=["health"], summary="Liveness probe")
def health() -> dict[str, str]:
    return {"status": "ok"}
