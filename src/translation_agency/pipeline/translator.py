"""Initial translation module using textgenhub with formatting preservation support."""

from textgenhub.chatgpt import ask
from ..config import TranslationAgencyConfig, get_prompt
from ..utils.file_handler import DocumentHandler, DocumentStructure, FormattedParagraph, FormattedTable
from pathlib import Path
from typing import Tuple, List, Union, Optional
import re
import time
import logging

logger = logging.getLogger(__name__)


class TranslationModule:
    """Handles initial document translation with optional formatting preservation."""
    
    def __init__(self, config: TranslationAgencyConfig):
        """
        Initialize translation module.
        
        Args:
            config: Configuration object containing LLM and pipeline settings
        """
        self.config = config
        self.max_llm_retries = 3
        self.llm_retry_backoff_ms = 2000  # Start with 2 seconds
    
    def _call_llm(self, prompt: str) -> str:
        """Call the ChatGPT LLM via textgenhub with retry logic and exponential backoff."""
        last_error = None
        
        for attempt in range(self.max_llm_retries):
            try:
                return ask(
                    prompt,
                    headless=self.config.llm.headless,
                    remove_cache=self.config.llm.remove_cache
                )
            except RuntimeError as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Only retry on session/login-related errors or JSON parse errors
                is_session_error = (
                    "login" in error_msg or 
                    "session" in error_msg or 
                    "json" in error_msg or
                    "did not produce json" in error_msg
                )
                
                if is_session_error and attempt < self.max_llm_retries - 1:
                    wait_time = (self.llm_retry_backoff_ms / 1000) * (2 ** attempt)
                    logger.warning(
                        f"LLM call failed (attempt {attempt+1}/{self.max_llm_retries}): {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    # Don't retry on non-session errors or after max retries
                    break
        
        raise RuntimeError(f"LLM call failed after {self.max_llm_retries} retries: {last_error}")
    
    
    def translate(self, input_path: str, preserve_formatting: bool = True, pipeline_timestamp: Optional[str] = None) -> Tuple[Union[str, DocumentStructure], str]:
        """
        Translate document from input path.
        
        Args:
            input_path: Path to the input document
            preserve_formatting: Whether to preserve formatting (DOCX only)
            
        Returns:
            Tuple of (translated_content, original_format)
            - translated_content is a DocumentStructure if preserve_formatting=True and format is DOCX
            - translated_content is a string otherwise
        """
        file_format = Path(input_path).suffix.lower()
        
        # For DOCX with formatting preservation
        if preserve_formatting and file_format == '.docx':
            return self._translate_docx_with_formatting(input_path)
        
        # For other formats, use plain text translation
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
            input_path, "step1_initial_translation", self.config.pipeline.output_dir, pipeline_timestamp
        )
        DocumentHandler.write_document(
            translated_content, output_path, file_format
        )
        
        return translated_content, file_format
    
    def _translate_docx_with_formatting(self, input_path: str) -> Tuple[DocumentStructure, str]:
        """
        Translate DOCX while preserving formatting.
        
        Args:
            input_path: Path to input DOCX file
            
        Returns:
            Tuple of (DocumentStructure with translated content, file format)
        """
        # Read document with formatting preserved
        structure, file_format = DocumentHandler.read_document_with_formatting(input_path)
        
        # Extract all text segments and translate them
        translated_texts = self._translate_structure(structure)
        
        # Save intermediate result with formatting preserved
        output_path = DocumentHandler.get_output_filename(
            input_path, "step1_initial_translation", self.config.pipeline.output_dir, pipeline_timestamp
        )
        DocumentHandler.write_document_with_formatting(
            structure, translated_texts, output_path, file_format
        )
        
        # Update structure with translated content for further validation steps
        self._update_structure_with_translations(structure, translated_texts)
        
        return structure, file_format
    
    def _translate_structure(self, structure: DocumentStructure) -> List[str]:
        """
        Translate all text segments in a document structure.
        Send them in batches for efficiency, but reconstruct properly.
        
        Args:
            structure: DocumentStructure object
            
        Returns:
            List of translated text segments
        """
        translated_texts = []
        
        # Collect all text items
        text_items = []
        for item in structure.paragraphs:
            if isinstance(item, FormattedParagraph):
                text = item.get_text()
                text_items.append((text, "paragraph"))
            elif isinstance(item, FormattedTable):
                for row in item.rows:
                    for cell_text in row:
                        text_items.append((cell_text, "cell"))
        
        # Batch into smaller chunks: max 10 items OR 300 words per batch
        # This is smaller than before (500 words) to help ChatGPT focus and not lose items
        batches = self._create_batches(text_items, max_words=300, max_items=10)
        
        # Translate each batch
        for batch in batches:
            # Separate empty from non-empty
            non_empty = [(text, item_type) for text, item_type in batch if text.strip()]
            empty = [(text, item_type) for text, item_type in batch if not text.strip()]
            
            if not non_empty:
                translated_texts.extend([text for text, _ in batch])
                continue
            
            # Create batch text as a simple list with line numbers
            batch_lines = []
            for idx, (text, _) in enumerate(non_empty):
                batch_lines.append(f"{idx}. {text}")
            
            batch_text = "\n\n".join(batch_lines)
            
            # Translate with instruction to maintain numbering
            prompt = f"""Translate the following numbered items to {self.config.pipeline.target_language}. Keep the exact same numbering format. Only translate the text after the number, preserve the number exactly.

{batch_text}"""
            
            translated_batch = self._call_llm(prompt)
            
            # Parse response and extract translated items
            translated_items = self._parse_numbered_response(translated_batch, len(non_empty))
            
            for translated in translated_items:
                translated_texts.append(translated)
            
            # Add empty items
            translated_texts.extend([text for text, _ in empty])
        
        return translated_texts
    
    def _parse_numbered_response(self, response: str, expected_count: int) -> List[str]:
        """Parse response from LLM - split by various delimiters to extract translated items."""
        import re
        
        # Skip intro phrases at the start
        skip_phrases = [
            'here is the translation',
            'the translation is',
            'here are the',
            'the following is',
        ]
        
        # Remove intro sentences
        for phrase in skip_phrases:
            idx = response.lower().find(phrase.lower())
            if idx == 0:  # Only at the very start
                # Find the end of this sentence (next newline)
                end_idx = response.find('\n', idx)
                if end_idx > 0:
                    response = response[end_idx:].lstrip()
        
        results = []
        
        # Strategy 1: Try numbered format (0. item, 1. item, etc)
        numbered = re.findall(r'^\d+\.\s*(.+?)(?=^\d+\.|$)', response, re.MULTILINE | re.DOTALL)
        if len(numbered) >= expected_count * 0.8:
            results = [item.strip() for item in numbered if item.strip()]
            logger.debug(f"Parsed {len(results)} items using numbered format")
            return results[:expected_count]
        
        # Strategy 2: Split by triple newlines (common ChatGPT format)
        if '\n\n\n' in response:
            parts = response.split('\n\n\n')
            filtered = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
            if len(filtered) >= expected_count * 0.7:
                logger.debug(f"Parsed {len(filtered)} items using triple-newline split")
                return filtered[:expected_count]
        
        # Strategy 3: Split by double newlines
        if '\n\n' in response:
            parts = response.split('\n\n')
            filtered = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
            if len(filtered) >= expected_count * 0.7:
                logger.debug(f"Parsed {len(filtered)} items using double-newline split")
                return filtered[:expected_count]
        
        # Strategy 4: Split by single newlines
        lines = response.split('\n')
        filtered = [line.strip() for line in lines if line.strip() and len(line.strip()) > 3]
        logger.debug(f"Parsed {len(filtered)} items using single-newline split")
        
        results = filtered
        
        # Pad with empty strings if needed
        while len(results) < expected_count:
            results.append("")
        
        return results[:expected_count]
    
    def _create_batches(self, items: List[Tuple[str, str]], max_words: int = 500, max_items: int = None) -> List[List[Tuple[str, str]]]:
        """
        Create batches of items that don't exceed max_words per batch and/or max items per batch.
        
        Args:
            items: List of (text, type) tuples
            max_words: Maximum words per batch
            max_items: Maximum items per batch (None = no limit)
            
        Returns:
            List of batches
        """
        batches = []
        current_batch = []
        current_word_count = 0
        
        for item in items:
            text, item_type = item
            word_count = len(text.split())
            
            # Check if we should start a new batch
            should_start_new_batch = False
            
            if max_items is not None and len(current_batch) >= max_items:
                should_start_new_batch = True
            elif item_type == "empty":
                current_batch.append(item)
                continue
            elif current_word_count + word_count > max_words:
                should_start_new_batch = True
            
            if should_start_new_batch and current_batch:
                batches.append(current_batch)
                current_batch = [item]
                current_word_count = word_count
            else:
                current_batch.append(item)
                current_word_count += word_count
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _update_structure_with_translations(self, structure: DocumentStructure, translated_texts: List[str]) -> None:
        """Update structure paragraphs with translated text."""
        text_index = 0
        
        for item in structure.paragraphs:
            if isinstance(item, FormattedParagraph):
                if text_index < len(translated_texts):
                    item.set_text(translated_texts[text_index])
                    text_index += 1
            elif isinstance(item, FormattedTable):
                for row_idx in range(len(item.rows)):
                    for col_idx in range(len(item.rows[row_idx])):
                        if text_index < len(translated_texts):
                            item.rows[row_idx][col_idx] = translated_texts[text_index]
                            text_index += 1

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
