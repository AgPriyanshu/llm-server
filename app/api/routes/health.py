"""Health check endpoint."""

from fastapi import APIRouter

from app.core import settings
from app.models import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint for load balancers and orchestrators."""
    return HealthResponse(status="healthy", version=settings.app_version)

