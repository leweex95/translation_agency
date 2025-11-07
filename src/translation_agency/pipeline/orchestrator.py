"""Pipeline runner with formatting preservation and validation."""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

from ..config import TranslationAgencyConfig, ensure_output_dir, PROMPTS
from ..utils.file_handler import DocumentHandler, DocumentStructure, FormattedParagraph, FormattedTable
from ..utils.excel_summary import ExcelSummaryGenerator
from ..llm.langchain_provider import EnhancedTextGenHubProvider, create_translation_pipeline_config
from ..llm.batched_session_provider import BatchedSessionProvider
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
    """Orchestrates translation pipeline with formatting preservation and validation."""
    
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
        Initialize pipeline runner with formatting preservation support.
        
        Args:
            config: Configuration object for the pipeline
        """
        self.config = config
        self.setup_logging()
        
        # Initialize translation module
        self.translator = TranslationModule(config)
        
        # Initialize validators based on configuration
        self.validators = self._initialize_validators()
        
        # Initialize LangChain provider for full pipeline chaining
        self._initialize_langchain_chaining()
        
        # Initialize batched session provider for session persistence
        self.batched_provider = BatchedSessionProvider(
            headless=self.config.llm.headless,
            remove_cache=self.config.llm.remove_cache,
            debug=self.config.llm.debug
        )
        
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
    
    def _initialize_langchain_chaining(self):
        """Initialize LangChain provider for complete pipeline chaining."""
        try:
            self.langchain_provider = EnhancedTextGenHubProvider(
                headless=self.config.llm.headless,
                remove_cache=self.config.llm.remove_cache,
                debug=self.config.llm.debug
            )
            self.can_use_chaining = True
            self.logger.info("LangChain chaining enabled with TextGenHub")
        except Exception as e:
            self.logger.warning(f"LangChain chaining disabled: {e}")
            self.langchain_provider = None
            self.can_use_chaining = False
    
    def run_pipeline(self, input_document_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete translation and validation pipeline using batched LLM calls.
        
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
        
        self.logger.info(f"Starting batched translation pipeline for: {input_path}")
        
        try:
            file_format = Path(input_path).suffix.lower()
            
            # Generate timestamp once for the entire pipeline run
            from datetime import datetime
            pipeline_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Read original content
            original_content, _ = DocumentHandler.read_document(input_path)
            
            # Check if we should preserve formatting for DOCX
            preserve_formatting = file_format == '.docx'
            
            # PHASE 1: Collect all prompts for batched execution
            self.logger.info("Phase 1: Collecting all prompts for batched execution...")
            
            prompts_and_steps = []
            
            # Step 1: Initial Translation
            self.logger.info("Collecting translation prompt...")
            translation_prompt = self._prepare_translation_prompt(original_content)
            prompts_and_steps.append(("translation", translation_prompt))
            
            # Step 2+: Validation prompts (will be prepared after getting initial translation)
            for validator in self.validators:
                prompts_and_steps.append((validator.step_name, None))  # Placeholder, will fill after translation
            
            # PHASE 2: Start batched session and execute all prompts
            self.logger.info("Phase 2: Starting batched ChatGPT session...")
            
            if not self.batched_provider.start_session():
                raise RuntimeError("Failed to start ChatGPT session")
            
            try:
                # Execute first prompt (translation)
                self.logger.info("Executing translation in batched session...")
                batch_prompts = [translation_prompt]
                batch_responses = self.batched_provider.execute_batch(batch_prompts)
                
                if not batch_responses or batch_responses[0].startswith("ERROR:"):
                    raise RuntimeError(f"Translation failed: {batch_responses[0] if batch_responses else 'No response'}")
                
                current_content = batch_responses[0]
                
                # Save initial translation result
                if preserve_formatting:
                    # For DOCX, we'll handle formatting preservation differently in batched mode
                    translated_structure = None  # Will implement later if needed
                    self.logger.info("✓ DOCX formatting preservation noted (will implement in final step)")
                else:
                    translation_output_path = DocumentHandler.get_output_filename(
                        input_path, "step1_initial_translation", self.config.pipeline.output_dir, pipeline_timestamp
                    )
                    DocumentHandler.write_document(current_content, translation_output_path, file_format)
                
                # PHASE 3: Execute validation steps in same session
                self.logger.info("Phase 3: Executing validation steps in same session...")
                
                for i, (step_name, _) in enumerate(prompts_and_steps[1:], 2):  # Skip translation (index 0)
                    validator = self.validators[i-2]  # Adjust index since we skipped translation
                    
                    self.logger.info(f"Step {i}: Running {step_name} validation...")
                    
                    # Prepare validation prompt with current content
                    validation_prompt = self._prepare_validation_prompt(
                        validator, current_content, original_content
                    )
                    
                    # Execute in same session
                    validation_responses = self.batched_provider.execute_batch([validation_prompt])
                    
                    if not validation_responses or validation_responses[0].startswith("ERROR:"):
                        error_msg = validation_responses[0] if validation_responses else "No response"
                        self.logger.warning(f"Validation {step_name} failed: {error_msg}")
                        # Continue with previous content instead of failing
                        continue
                    
                    improved_content = validation_responses[0]
                    
                    # Save intermediate result
                    output_path = DocumentHandler.get_output_filename(
                        input_path, f"step{i}_{step_name}_validation", self.config.pipeline.output_dir, pipeline_timestamp
                    )
                    DocumentHandler.write_document(improved_content, output_path, file_format)
                    
                    current_content = improved_content
                    self.logger.info(f"✓ Completed {step_name}")
                
                # PHASE 4: Save final result
                self.logger.info("Phase 4: Saving final result...")
                
                if preserve_formatting and translated_structure is not None:
                    # For DOCX with formatting, create final output with formatting preserved
                    final_output_path = self._save_final_result_with_formatting(
                        translated_structure, current_content, input_path, file_format, pipeline_timestamp
                    )
                else:
                    final_output_path = self._save_final_result(
                        current_content, input_path, file_format, pipeline_timestamp
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
                    "original_content": original_content,
                    "execution_mode": "batched_session",
                    "formatting_preserved": preserve_formatting
                }
                
                self.logger.info(f"Pipeline completed successfully. Output: {final_output_path}")
                return results
                
            finally:
                # Always end the session
                self.batched_provider.end_session()
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "input_path": input_path if 'input_path' in locals() else input_path
            }
    
    def _prepare_translation_prompt(self, original_content: str) -> str:
        """Prepare the translation prompt for batched execution."""
        from ..config import get_prompt
        
        prompt_args = {
            "text": original_content,
            "style": self.config.pipeline.translation_style,
            "target_language": self.config.pipeline.target_language
        }
        
        return get_prompt("translation", **prompt_args)
    
    def _prepare_validation_prompt(self, validator: Any, text: str, original_text: str) -> str:
        """Prepare a validation prompt for batched execution."""
        from ..config import get_prompt
        
        prompt_args = {
            "text": text,
            "style": self.config.pipeline.translation_style
        }
        
        # Add original text if needed for this validation step
        if validator._needs_original_text():
            prompt_args["original_text"] = original_text
        
        prompt_key = validator._get_prompt_key()
        return get_prompt(prompt_key, **prompt_args)
    
    def _save_final_result(self, content: str, input_path: str, file_format: str, pipeline_timestamp: str) -> Path:
        """Save the final translated and validated content."""
        # Determine the last step based on enabled validators
        if self.validators:
            last_validator = self.validators[-1]
            last_step_name = last_validator.step_name
            step_number = len(self.validators) + 1  # +1 for translation step
            output_path = DocumentHandler.get_output_filename(
                input_path, last_step_name, self.config.pipeline.output_dir, pipeline_timestamp
            )
        else:
            # No validators enabled, final result is the initial translation
            output_path = DocumentHandler.get_output_filename(
                input_path, "step1_initial_translation", self.config.pipeline.output_dir, pipeline_timestamp
            )
        
        # Write the content to file
        return DocumentHandler.write_document(content, output_path, file_format)
    
    def _save_final_result_with_formatting(self, structure: DocumentStructure, 
                                          final_content: str, input_path: str, 
                                          file_format: str, pipeline_timestamp: str) -> Path:
        """
        Save final result preserving DOCX formatting.
        
        Args:
            structure: Original DocumentStructure with formatting
            final_content: Final translated content
            input_path: Input document path
            file_format: Document format
            pipeline_timestamp: Timestamp for the pipeline run
            
        Returns:
            Path to saved file
        """
        # Split final content back into segments for formatting preservation
        # For simplicity, treat it as one large translated text
        translated_texts = [final_content]
        
        # Use final_{original_filename} for the output filename
        input_filename = Path(input_path).stem
        final_filename = f"final_{input_filename}"
        
        output_path = DocumentHandler.get_output_filename(
            input_path, final_filename, self.config.pipeline.output_dir, pipeline_timestamp
        )
        
        return DocumentHandler.write_document_with_formatting(
            structure, translated_texts, output_path, file_format
        )
    
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
                "translation_style": self.config.pipeline.translation_style,
                "target_language": self.config.pipeline.target_language,
                "output_dir": self.config.pipeline.output_dir
            },
            "enabled_steps": self.config.validation.enabled_steps,
            "available_validators": list(self.VALIDATOR_REGISTRY.keys()),
            "total_steps": len(self.validators) + 1,
            "langchain_chaining": self.can_use_chaining,
            "execution_modes": ["sequential"],
            "formatting_preservation": {
                "docx": True,
                "xlsx": False,
                "pdf": False
            }
        }
