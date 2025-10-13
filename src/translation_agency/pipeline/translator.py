"""Initial translation module using textgenhub."""

from textgenhub import ChatGPT, ask_chatgpt
from ..config import TranslationAgencyConfig, get_prompt
from ..utils.file_handler import DocumentHandler
from pathlib import Path
from typing import Tuple


class TranslationModule:
    """Handles initial document translation."""
    
    def __init__(self, config: TranslationAgencyConfig):
        """
        Initialize translation module.
        
        Args:
            config: Configuration object containing LLM and pipeline settings
        """
        self.config = config
        self.llm_provider = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM provider based on configuration."""
        backend = self.config.llm.backend
        
        if backend == "chatgpt":
            return ChatGPT(
                headless=self.config.llm.headless,
                remove_cache=self.config.llm.remove_cache
            )
        else:
            raise ValueError(f"Only ChatGPT backend is currently supported. Got: {backend}")
    
    def translate(self, input_path: str) -> Tuple[str, str]:
        """
        Translate document from input path.
        
        Args:
            input_path: Path to the input document
            
        Returns:
            Tuple of (translated_content, original_format)
        """
        # Read the input document
        content, file_format = DocumentHandler.read_document(input_path)
        
        # Prepare translation prompt
        prompt = get_prompt(
            "translation",
            target_language=self.config.pipeline.target_language,
            style=self.config.pipeline.translation_style,
            text=content
        )
        
        # Perform translation
        translated_content = self._call_llm(prompt)
        
        # Save intermediate result
        output_path = DocumentHandler.get_output_filename(
            input_path, "step1_initial_translation", self.config.pipeline.output_dir
        )
        DocumentHandler.write_document(
            translated_content, output_path, file_format
        )
        
        return translated_content, file_format
    
    def _call_llm(self, prompt: str) -> str:
        """Call the appropriate LLM backend."""
        backend = self.config.llm.backend
        
        if backend == "chatgpt":
            return self.llm_provider.chat(prompt)
        else:
            raise ValueError(f"Unsupported backend: {backend}. Only ChatGPT is currently supported.")

    def translate_text(self, text: str) -> str:
        """
        Translate raw text content.
        
        Args:
            text: Text content to translate
            
        Returns:
            Translated text
        """
        prompt = get_prompt(
            "translation",
            target_language=self.config.pipeline.target_language,
            style=self.config.pipeline.translation_style,
            text=text
        )
        
        return self._call_llm(prompt)