"""Main entry point for the translation agency pipeline."""

import argparse
import sys
from pathlib import Path
from typing import Optional
import json

from .config import TranslationAgencyConfig
from .pipeline.orchestrator import PipelineRunner


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command line interface parser."""
    parser = argparse.ArgumentParser(
        description="AI-powered modular translation pipeline with multi-step validation"
    )
    
    parser.add_argument(
        "input_document",
        help="Path to the document to translate (.txt, .md, .docx, .pdf)"
    )
    
    parser.add_argument(
        "--style",
        required=True,
        choices=["professional", "casual", "academic", "creative", "technical"],
        help="Translation style (required)"
    )
    
    parser.add_argument(
        "--target-language",
        required=True,
        help="Target language (required)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "--llm-backend",
        choices=["chatgpt"],
        default="chatgpt",
        help="LLM backend to use (currently only ChatGPT is supported)"
    )
    
    parser.add_argument(
        "--headless",
        type=lambda x: str(x).lower() in ['true', '1', 'yes', 'on'],
        default=True,
        help="Run browser in headless mode (default: true). Use --headless false to show browser."
    )
    
    parser.add_argument(
        "--debug",
        action="store_true", 
        help="Enable debug mode with detailed logging"
    )
    
    parser.add_argument(
        "--disable-steps",
        nargs="*",
        choices=["grammar", "style", "accuracy", "hallucination", "consistency", "crossllm"],
        help="Validation steps to disable"
    )
    

    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show pipeline status and exit"
    )
    
    return parser


def main():
    """Main entry point for CLI."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Load configuration from environment
    config = TranslationAgencyConfig.from_env()
    
    # Override config with CLI arguments
    config.pipeline.input_document = args.input_document
    config.pipeline.translation_style = args.style
    config.pipeline.target_language = args.target_language
    config.pipeline.output_dir = args.output_dir
    config.llm.backend = args.llm_backend
    config.llm.headless = args.headless
    config.llm.debug = args.debug
    
    # Disable specified validation steps
    if args.disable_steps:
        enabled_steps = [
            step for step in config.validation.enabled_steps 
            if step not in args.disable_steps
        ]
        config.validation.enabled_steps = enabled_steps
    
    # Initialize pipeline runner
    pipeline = PipelineRunner(config)
    
    # Show status if requested
    if args.status:
        status = pipeline.get_pipeline_status()
        print(json.dumps(status, indent=2))
        return
    
    # Validate input file exists
    if not Path(args.input_document).exists():
        print(f"Error: Input file not found: {args.input_document}")
        sys.exit(1)
    
    # Run the pipeline
    print(f"Starting translation pipeline for: {args.input_document}")
    results = pipeline.run_pipeline()
    
    if results["success"]:
        print(f"✅ Translation completed successfully!")
        print(f"📄 Input: {results['input_path']}")
        print(f"📁 Output: {results['output_path']}")
        print(f"🔧 Steps completed: {results['steps_completed']}")
    else:
        print(f"❌ Translation failed: {results['error']}")
        sys.exit(1)


def run_pipeline_programmatic(
    input_document: str,
    style: str,
    target_language: str, 
    output_dir: str = "output",
    llm_backend: str = "chatgpt",
    headless: bool = True,
    debug: bool = False,
    disabled_steps: Optional[list] = None
) -> dict:
    """
    Run pipeline programmatically without CLI.
    
    Args:
        input_document: Path to input document
        style: Translation style
        target_language: Target language  
        source_language: Source language
        output_dir: Output directory
        llm_backend: LLM backend
        model: LLM model
        temperature: LLM temperature
        disabled_steps: List of steps to disable
        
    Returns:
        Dictionary with results
    """
    # Load base configuration
    config = TranslationAgencyConfig.from_env()
    
    # Apply parameters
    config.pipeline.input_document = input_document
    config.pipeline.translation_style = style
    config.pipeline.target_language = target_language
    config.pipeline.output_dir = output_dir
    config.llm.backend = llm_backend
    config.llm.headless = headless
    config.llm.debug = debug
    
    # Handle disabled steps
    if disabled_steps:
        enabled_steps = [
            step for step in config.validation.enabled_steps 
            if step not in disabled_steps
        ]
        config.validation.enabled_steps = enabled_steps
    
    # Run pipeline
    pipeline = PipelineRunner(config)
    return pipeline.run_pipeline()


if __name__ == "__main__":
    main()