"""Configuration management for the translation pipeline."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for LLM backends."""
    backend: str = "chatgpt"  # chatgpt, deepseek, perplexity
    headless: bool = True
    remove_cache: bool = True
    debug: bool = False


@dataclass
class PipelineConfig:
    """Configuration for the translation pipeline."""
    input_document: str = ""
    translation_style: str = "professional"  # Required, no default
    target_language: str = ""  # Required, no default
    output_dir: str = "output"


@dataclass
class ValidationConfig:
    """Configuration for validation steps."""
    enabled_steps: List[str] = field(default_factory=lambda: [
        "grammar",
        "style",
        "accuracy",
        "hallucination",
        "consistency",
        "crossllm"
    ])
    crossllm_backend: str = "chatgpt"  # Same LLM for cross-validation (for now)


@dataclass
class TranslationAgencyConfig:
    """Main configuration class."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    @classmethod
    def from_env(cls) -> "TranslationAgencyConfig":
        """Load configuration from environment variables."""
        config = cls()

        # LLM Configuration
        config.llm.backend = os.getenv("LLM_BACKEND", config.llm.backend)
        config.llm.headless = os.getenv("HEADLESS", "true").lower() == "true"
        config.llm.remove_cache = os.getenv("REMOVE_CACHE", "true").lower() == "true"
        config.llm.debug = os.getenv("DEBUG_MODE", "false").lower() == "true"

        # Pipeline Configuration
        config.pipeline.input_document = os.getenv("INPUT_DOCUMENT", config.pipeline.input_document)
        config.pipeline.output_dir = os.getenv("OUTPUT_DIR", config.pipeline.output_dir)

        # Validation Configuration
        config.validation.crossllm_backend = os.getenv("CROSSLLM_BACKEND", config.validation.crossllm_backend)

        return config


def load_prompt(prompt_name: str) -> str:
    """Load a prompt from the prompts directory."""
    import importlib
    try:
        module = importlib.import_module(f"prompts.{prompt_name}")
        var_name = f"{prompt_name.upper()}_PROMPT"
        return getattr(module, var_name)
    except ImportError as e:
        raise ImportError(f"Could not import prompt module: prompts.{prompt_name} - {e}")
    except AttributeError as e:
        raise AttributeError(f"Could not find prompt variable in prompts.{prompt_name} - {e}")


# Predefined prompts for each validation step
PROMPTS = {
    "translation": load_prompt("translation"),
    "grammar": load_prompt("grammar"),
    "style": load_prompt("style"),
    "accuracy": load_prompt("accuracy"),
    "hallucination": load_prompt("hallucination"),
    "consistency": load_prompt("consistency"),
    "crossllm": load_prompt("crossllm")
}


def get_prompt(step: str, **kwargs) -> str:
    """Get formatted prompt for a validation step."""
    if step not in PROMPTS:
        raise ValueError(f"Unknown validation step: {step}")

    return PROMPTS[step].format(**kwargs)


def ensure_output_dir(output_dir: str) -> Path:
    """Ensure output directory exists and return Path object."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path