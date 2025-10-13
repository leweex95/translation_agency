"""Translation Agency - AI-powered modular translation pipeline."""

from .config import TranslationAgencyConfig, PROMPTS
from .pipeline.orchestrator import PipelineRunner
from .pipeline.translator import TranslationModule
from .utils.file_handler import DocumentHandler
from .pipeline.validators import (
    GrammarValidator,
    StyleValidator,
    AccuracyValidator,
    HallucinationValidator,
    ConsistencyValidator,
    CrossLLMValidator
)
from .main import run_pipeline_programmatic

__version__ = "0.1.0"
__all__ = [
    "TranslationAgencyConfig",
    "PipelineRunner",
    "TranslationModule",
    "DocumentHandler",
    "GrammarValidator",
    "StyleValidator",
    "AccuracyValidator",
    "HallucinationValidator",
    "ConsistencyValidator",
    "CrossLLMValidator",
    "run_pipeline_programmatic",
    "PROMPTS"
]