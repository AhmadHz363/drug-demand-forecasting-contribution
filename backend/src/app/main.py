import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.hospital_ingestion import router as hospital_ingestion_router
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
        {"name": "Forecasting", "description": "SHIELD-XR training and performance monitoring."},
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

app.include_router(auth_router)
app.include_router(hospital_ingestion_router)
app.include_router(shield_xr_forecasting_router)


@app.get("/health", tags=["health"], summary="Liveness probe")
def health() -> dict[str, str]:
    return {"status": "ok"}
