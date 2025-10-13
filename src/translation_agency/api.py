"""FastAPI application for n8n integration and programmatic access to translation pipeline."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
from pathlib import Path
import json
from datetime import datetime

from .main import run_pipeline_programmatic
from .config import TranslationAgencyConfig

app = FastAPI(
    title="Translation Agency API",
    description="AI-powered modular translation pipeline with multi-step validation",
    version="1.0.0"
)

class TranslationRequest(BaseModel):
    """Request model for translation pipeline."""
    input_document: str
    translation_style: str = "professional"
    target_language: str = "hungarian"
    source_language: Optional[str] = None
    output_dir: str = "output"
    llm_backend: str = "chatgpt"
    headless: bool = True
    debug: bool = False
    disabled_steps: Optional[list] = None

class TranslationResponse(BaseModel):
    """Response model for translation pipeline."""
    success: bool
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    file_format: Optional[str] = None
    steps_completed: Optional[int] = None
    final_content: Optional[str] = None
    original_content: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime
    execution_time: Optional[float] = None

@app.post("/translate", response_model=TranslationResponse)
async def translate_document(request: TranslationRequest, background_tasks: BackgroundTasks):
    """
    Execute translation pipeline.

    This endpoint can be called by n8n workflows or other automation tools.
    """
    start_time = datetime.now()

    try:
        # Validate input file exists
        if not Path(request.input_document).exists():
            raise HTTPException(
                status_code=400,
                detail=f"Input file not found: {request.input_document}"
            )

        # Run pipeline programmatically
        result = run_pipeline_programmatic(
            input_document=request.input_document,
            style=request.translation_style,
            target_language=request.target_language,
            output_dir=request.output_dir,
            llm_backend=request.llm_backend,
            headless=request.headless,
            debug=request.debug,
            disabled_steps=request.disabled_steps
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        return TranslationResponse(
            **result,
            timestamp=datetime.now(),
            execution_time=execution_time
        )

    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }

@app.get("/status")
async def get_pipeline_status():
    """Get current pipeline configuration and status."""
    try:
        config = TranslationAgencyConfig.from_env()
        return {
            "status": "configured",
            "config": {
                "llm_backend": config.llm.backend,
                "validation_steps": config.validation.enabled_steps,
                "output_dir": config.pipeline.output_dir
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Configuration error: {str(e)}"
        )

@app.post("/webhook/n8n")
async def n8n_webhook(request: Dict[str, Any]):
    """
    n8n webhook endpoint for workflow integration.

    Expects JSON payload with translation parameters.
    """
    try:
        # Extract parameters from n8n payload
        data = request.get("json", {})

        translation_request = TranslationRequest(
            input_document=data.get("input_document", ""),
            translation_style=data.get("translation_style", "professional"),
            target_language=data.get("target_language", "hungarian"),
            source_language=data.get("source_language"),
            output_dir=data.get("output_dir", "output/n8n"),
            llm_backend=data.get("llm_backend", "chatgpt"),
            headless=data.get("headless", True),
            debug=data.get("debug", False),
            disabled_steps=data.get("disabled_steps")
        )

        # Execute translation
        result = run_pipeline_programmatic(
            input_document=translation_request.input_document,
            style=translation_request.translation_style,
            target_language=translation_request.target_language,
            output_dir=translation_request.output_dir,
            llm_backend=translation_request.llm_backend,
            headless=translation_request.headless,
            debug=translation_request.debug,
            disabled_steps=translation_request.disabled_steps
        )

        # Return n8n-compatible response
        return {
            "success": result["success"],
            "pipeline_success": result["success"],
            "pipeline_output": result.get("output_path", ""),
            "pipeline_error": result.get("error", ""),
            "input_document": translation_request.input_document,
            "output_path": result.get("output_path", ""),
            "steps_completed": result.get("steps_completed", 0),
            "timestamp": datetime.now().isoformat(),
            "execution_details": {
                "file_format": result.get("file_format"),
                "final_content_length": len(result.get("final_content", "")),
                "original_content_length": len(result.get("original_content", ""))
            }
        }

    except Exception as e:
        return {
            "success": False,
            "pipeline_success": False,
            "pipeline_error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

def run_api():
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)