"""Base validation class with LangChain integration support."""

from abc import ABC, abstractmethod
from textgenhub.chatgpt import ask
from ..config import TranslationAgencyConfig, get_prompt
from ..utils.file_handler import DocumentHandler
from ..llm.langchain_provider import EnhancedTextGenHubProvider
from typing import Optional


class BaseValidator(ABC):
    """Base class for all validation steps with LangChain integration."""

    def __init__(self, config: TranslationAgencyConfig):
        """
        Initialize validator with dual provider support.

        Args:
            config: Configuration object containing LLM and pipeline settings
        """
        self.config = config
        self.step_name = self._get_step_name()
        
        # Initialize both providers for comparison/fallback
        self.textgenhub_provider = self._initialize_textgenhub()
        
        try:
            self.langchain_provider = EnhancedTextGenHubProvider(
                headless=self.config.llm.headless,
                remove_cache=self.config.llm.remove_cache,
                debug=self.config.llm.debug,
                fallback_provider=self.textgenhub_provider
            )
            self.use_langchain = True
        except Exception as e:
            print(f"[Warning] LangChain initialization failed: {e}")
            self.langchain_provider = None
            self.use_langchain = False

    def _initialize_textgenhub(self):
        """Initialize the textgenhub provider (existing)."""
        backend = self.config.llm.backend

        if backend == "chatgpt":
            # Return a simple wrapper that calls ask function
            class TextGenHubWrapper:
                def __init__(self, config):
                    self.config = config
                
                def chat(self, prompt):
                    return ask(
                        prompt,
                        headless=self.config.llm.headless,
                        remove_cache=self.config.llm.remove_cache
                    )
            
            return TextGenHubWrapper(self.config)
        else:
            raise ValueError(f"Only ChatGPT backend is currently supported. Got: {backend}")

    @abstractmethod
    def _get_step_name(self) -> str:
        """Return the name of this validation step."""
        pass

    @abstractmethod
    def _get_prompt_key(self) -> str:
        """Return the prompt key for this validation step."""
        pass

    def validate(self, text: str, original_text: Optional[str] = None, 
                input_path: Optional[str] = None, file_format: Optional[str] = None,
                pipeline_timestamp: Optional[str] = None) -> str:
        """
        Validate and improve the given text using available provider.

        Args:
            text: Text to validate and improve
            original_text: Original source text for accuracy checks
            input_path: Original input file path (for saving intermediate results)
            file_format: Original file format

        Returns:
            Improved text
        """
        # Prepare prompt arguments
        prompt_args = {
            "text": text,
            "style": self.config.pipeline.translation_style
        }

        # Add original text if needed for this validation step
        if original_text and self._needs_original_text():
            prompt_args["original_text"] = original_text

        # Get and format prompt
        prompt = get_prompt(self._get_prompt_key(), **prompt_args)

        # Perform validation using preferred provider
        improved_text = self._call_llm(prompt)

        # Save intermediate result
        if input_path and file_format:
            output_path = DocumentHandler.get_output_filename(
                input_path, self.step_name, self.config.pipeline.output_dir, pipeline_timestamp
            )
            DocumentHandler.write_document(improved_text, output_path, file_format)

        return improved_text

    def _call_llm(self, prompt: str) -> str:
        """Call the appropriate LLM backend with LangChain preference."""
        # Try LangChain first if available
        if self.use_langchain and self.langchain_provider:
            try:
                return self.langchain_provider.chat_with_fallback(prompt)
            except Exception as e:
                print(f"[Warning] LangChain failed for {self.step_name}: {e}")
                # Fall through to textgenhub
        
        # Fallback to textgenhub
        backend = self.config.llm.backend
        if backend == "chatgpt":
            return self.textgenhub_provider.chat(prompt)
        else:
            raise ValueError(f"Unsupported backend: {backend}. Only ChatGPT is currently supported.")

    def _needs_original_text(self) -> bool:
        """Return True if this validator needs access to original text."""
        return self._get_prompt_key() in ["accuracy", "hallucination"]