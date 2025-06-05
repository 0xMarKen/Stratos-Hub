"""
StratosHub Agent Runtime Service

Main FastAPI application for executing AI agents with comprehensive
monitoring, error handling, and performance optimization.
"""

import asyncio
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .core.config import get_settings
from .core.logging import setup_logging, get_logger
from .core.metrics import MetricsCollector
from .core.database import Database
from .core.cache import CacheManager
from .models.registry import ModelRegistry
from .execution.engine import ExecutionEngine
from .execution.scheduler import TaskScheduler
from .api.routes import agents, executions, models, health
from .middleware.auth import AuthMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.error_handler import ErrorHandlerMiddleware

# Initialize components
settings = get_settings()
logger = get_logger(__name__)
metrics = MetricsCollector()
db = Database()
cache = CacheManager()
model_registry = ModelRegistry()
execution_engine = ExecutionEngine()
scheduler = TaskScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting StratosHub Agent Runtime Service")
    
    try:
        # Initialize database connections
        await db.connect()
        logger.info("Database connection established")
        
        # Initialize cache
        await cache.connect()
        logger.info("Cache connection established")
        
        # Initialize model registry
        await model_registry.initialize()
        logger.info(f"Model registry initialized with {model_registry.count} models")
        
        # Start execution engine
        await execution_engine.start()
        logger.info("Execution engine started")
        
        # Start task scheduler
        await scheduler.start()
        logger.info("Task scheduler started")
        
        # Setup graceful shutdown
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating graceful shutdown")
            asyncio.create_task(shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("StratosHub Agent Runtime Service started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        sys.exit(1)
    
    # Shutdown
    logger.info("Shutting down StratosHub Agent Runtime Service")
    await shutdown()


async def shutdown():
    """Graceful shutdown sequence"""
    try:
        # Stop accepting new tasks
        await scheduler.stop()
        logger.info("Task scheduler stopped")
        
        # Wait for running executions to complete (with timeout)
        await execution_engine.shutdown(timeout=30)
        logger.info("Execution engine stopped")
        
        # Close database connections
        await db.disconnect()
        logger.info("Database connection closed")
        
        # Close cache connections
        await cache.disconnect()
        logger.info("Cache connection closed")
        
        logger.info("Graceful shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="StratosHub Agent Runtime",
    description="Enterprise-grade AI agent execution runtime on Solana blockchain",
    version="0.1.0",
    docs_url="/docs" if settings.environment == "development" else None,
    redoc_url="/redoc" if settings.environment == "development" else None,
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthMiddleware)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(executions.router, prefix="/api/v1/executions", tags=["executions"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])


@app.get("/metrics", include_in_schema=False)
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "StratosHub Agent Runtime",
        "version": "0.1.0",
        "status": "operational",
        "environment": settings.environment,
        "features": {
            "model_types": model_registry.supported_types,
            "max_concurrent_executions": execution_engine.max_concurrent,
            "available_models": model_registry.count,
        },
        "metrics": {
            "total_executions": metrics.get_counter("executions_total"),
            "active_executions": metrics.get_gauge("executions_active"),
            "average_execution_time": metrics.get_histogram("execution_duration").mean,
        }
    }


@app.get("/api/v1/status")
async def get_service_status():
    """Detailed service status for monitoring"""
    try:
        # Check component health
        db_healthy = await db.health_check()
        cache_healthy = await cache.health_check()
        engine_healthy = execution_engine.is_healthy()
        scheduler_healthy = scheduler.is_healthy()
        
        return {
            "status": "healthy" if all([db_healthy, cache_healthy, engine_healthy, scheduler_healthy]) else "degraded",
            "timestamp": asyncio.get_event_loop().time(),
            "components": {
                "database": "healthy" if db_healthy else "unhealthy",
                "cache": "healthy" if cache_healthy else "unhealthy",
                "execution_engine": "healthy" if engine_healthy else "unhealthy",
                "task_scheduler": "healthy" if scheduler_healthy else "unhealthy",
            },
            "resource_usage": {
                "cpu_percent": metrics.get_gauge("cpu_usage_percent"),
                "memory_percent": metrics.get_gauge("memory_usage_percent"),
                "disk_usage_percent": metrics.get_gauge("disk_usage_percent"),
            },
            "performance": {
                "requests_per_second": metrics.get_rate("requests_total"),
                "average_response_time": metrics.get_histogram("request_duration").mean,
                "error_rate": metrics.get_rate("errors_total"),
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    metrics.increment_counter("http_errors_total", {"status_code": exc.status_code})
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url),
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    metrics.increment_counter("unhandled_errors_total")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "path": str(request.url),
        }
    )


if __name__ == "__main__":
    # Setup logging
    setup_logging(
        level=settings.log_level,
        format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.environment == "development",
        workers=1 if settings.environment == "development" else settings.workers,
        log_level=settings.log_level.lower(),
        access_log=True,
        use_colors=True,
    ) 