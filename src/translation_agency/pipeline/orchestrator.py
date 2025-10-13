"""Pipeline runner for orchestrating translation and validation steps."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from ..config import TranslationAgencyConfig, ensure_output_dir
from ..utils.file_handler import DocumentHandler
from ..utils.excel_summary import ExcelSummaryGenerator
from .translator import TranslationModule
from .validators import (
    GrammarValidator,
    StyleValidator,
    AccuracyValidator,
    HallucinationValidator,
    ConsistencyValidator,
    CrossLLMValidator
)


class PipelineRunner:
    """Orchestrates the complete translation and validation pipeline."""
    
    # Registry of available validation steps
    VALIDATOR_REGISTRY = {
        "grammar": GrammarValidator,
        "style": StyleValidator,
        "accuracy": AccuracyValidator,
        "hallucination": HallucinationValidator,
        "consistency": ConsistencyValidator,
        "crossllm": CrossLLMValidator
    }
    
    def __init__(self, config: TranslationAgencyConfig):
        """
        Initialize pipeline runner.
        
        Args:
            config: Configuration object for the pipeline
        """
        self.config = config
        self.setup_logging()
        
        # Initialize translation module
        self.translator = TranslationModule(config)
        
        # Initialize validators based on configuration
        self.validators = self._initialize_validators()
        
        # Ensure output directory exists
        ensure_output_dir(self.config.pipeline.output_dir)
    
    def setup_logging(self):
        """Setup logging for pipeline execution."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_validators(self) -> List[Any]:
        """Initialize validator instances based on configuration."""
        validators = []
        
        for step_name in self.config.validation.enabled_steps:
            if step_name in self.VALIDATOR_REGISTRY:
                validator_class = self.VALIDATOR_REGISTRY[step_name]
                validators.append(validator_class(self.config))
                self.logger.info(f"Initialized validator: {step_name}")
            else:
                self.logger.warning(f"Unknown validation step: {step_name}")
        
        return validators
    
    def run_pipeline(self, input_document_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete translation and validation pipeline.
        
        Args:
            input_document_path: Path to input document (overrides config)
            
        Returns:
            Dictionary containing results and metadata
        """
        # Use provided path or config path
        input_path = input_document_path or self.config.pipeline.input_document
        
        if not input_path:
            raise ValueError("No input document specified in config or parameters")
        
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input document not found: {input_path}")
        
        self.logger.info(f"Starting translation pipeline for: {input_path}")
        
        try:
            # Step 1: Initial Translation
            self.logger.info("Step 1: Performing initial translation...")
            original_content, file_format = DocumentHandler.read_document(input_path)
            translated_content, file_format = self.translator.translate(input_path)
            
            # Store current content for validation steps
            current_content = translated_content
            
            # Step 2-7: Validation rounds
            for i, validator in enumerate(self.validators, 2):
                step_name = validator.step_name
                self.logger.info(f"Step {i}: Running {step_name} validation...")
                
                current_content = validator.validate(
                    text=current_content,
                    original_text=original_content,
                    input_path=input_path,
                    file_format=file_format
                )
                
                self.logger.info(f"Completed {step_name}")
            
            # Save final result
            final_output_path = self._save_final_result(
                current_content, input_path, file_format
            )
            
            # Generate Excel summary
            try:
                document_name = Path(input_path).stem
                excel_path = ExcelSummaryGenerator.generate_summary(
                    self.config.pipeline.output_dir, document_name
                )
                self.logger.info(f"Excel summary generated: {excel_path}")
            except Exception as e:
                self.logger.warning(f"Failed to generate Excel summary: {e}")
            
            results = {
                "success": True,
                "input_path": input_path,
                "output_path": str(final_output_path),
                "file_format": file_format,
                "steps_completed": len(self.validators) + 1,  # +1 for translation
                "final_content": current_content,
                "original_content": original_content
            }
            
            self.logger.info(f"Pipeline completed successfully. Output: {final_output_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "input_path": input_path
            }
    
    def _save_final_result(self, content: str, input_path: str, file_format: str) -> Path:
        """Save the final translated and validated content."""
        # Instead of creating a duplicate final.txt, return the path to the last step file
        # The final content is already saved as step7_crossllm_validation.txt (or the last validation step)
        last_step_num = len(self.validators) + 1  # +1 for translation step
        output_path = DocumentHandler.get_output_filename(
            input_path, f"step{last_step_num}_crossllm_validation", self.config.pipeline.output_dir
        )
        
        # For PDF inputs, the actual file has .txt extension
        if file_format == '.pdf':
            output_path = output_path.with_suffix('.txt')
        
        return output_path
    
    def run_single_step(self, step_name: str, text: str, 
                       original_text: Optional[str] = None) -> str:
        """
        Run a single validation step on provided text.
        
        Args:
            step_name: Name of the validation step
            text: Text to validate
            original_text: Original source text (for accuracy/hallucination steps)
            
        Returns:
            Validated text
        """
        if step_name == "translation":
            return self.translator.translate_text(text)
        
        if step_name not in self.VALIDATOR_REGISTRY:
            raise ValueError(f"Unknown step: {step_name}")
        
        validator_class = self.VALIDATOR_REGISTRY[step_name]
        validator = validator_class(self.config)
        
        return validator.validate(text, original_text)
    
    def add_custom_validator(self, step_name: str, validator_class: Any) -> None:
        """
        Add a custom validator to the registry.
        
        Args:
            step_name: Name for the validation step
            validator_class: Validator class (must inherit from BaseValidator)
        """
        self.VALIDATOR_REGISTRY[step_name] = validator_class
        self.logger.info(f"Added custom validator: {step_name}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline configuration status."""
        return {
            "config": {
                "llm_backend": self.config.llm.backend,
                "model": self.config.llm.model,
                "translation_style": self.config.pipeline.translation_style,
                "target_language": self.config.pipeline.target_language,
                "output_dir": self.config.pipeline.output_dir
            },
            "enabled_steps": self.config.validation.enabled_steps,
            "available_validators": list(self.VALIDATOR_REGISTRY.keys()),
            "total_steps": len(self.validators) + 1
        }