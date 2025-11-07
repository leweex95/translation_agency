"""Consolidated validation modules for the translation pipeline."""

from .base_validator import BaseValidator


class GrammarValidator(BaseValidator):
    """Validator for grammar and spelling checks."""

    def _get_step_name(self) -> str:
        return "step2_grammar_validation"

    def _get_prompt_key(self) -> str:
        return "grammar"


class StyleValidator(BaseValidator):
    """Validator for style and tone consistency."""

    def _get_step_name(self) -> str:
        return "step3_style_validation"

    def _get_prompt_key(self) -> str:
        return "style"


class AccuracyValidator(BaseValidator):
    """Validator for translation accuracy and faithfulness."""

    def _get_step_name(self) -> str:
        return "step4_accuracy_validation"

    def _get_prompt_key(self) -> str:
        return "accuracy"


class HallucinationValidator(BaseValidator):
    """Validator for detecting and correcting hallucinated content."""

    def _get_step_name(self) -> str:
        return "step5_hallucination_validation"

    def _get_prompt_key(self) -> str:
        return "hallucination"


class ConsistencyValidator(BaseValidator):
    """Validator for terminology and style consistency."""

    def _get_step_name(self) -> str:
        return "step6_consistency_validation"

    def _get_prompt_key(self) -> str:
        return "consistency"


class CrossLLMValidator(BaseValidator):
    """Validator using a different LLM for final validation."""

    def _initialize_llm(self):
        """Initialize alternative LLM provider for cross-validation."""
        backend = self.config.validation.crossllm_backend

        if backend == "chatgpt":
            # Use the same REAL textgenhub approach as other validators
            from textgenhub.chatgpt import ask
            
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
            raise ValueError(f"Only ChatGPT backend is currently supported for cross-validation. Got: {backend}")

    def _call_llm(self, prompt: str) -> str:
        """Call the appropriate cross-LLM backend."""
        backend = self.config.validation.crossllm_backend

        if backend == "chatgpt":
            return self.llm_provider.chat(prompt)
        else:
            raise ValueError(f"Unsupported backend: {backend}. Only ChatGPT is currently supported.")

    def _get_step_name(self) -> str:
        return "step7_crossllm_validation"

    def _get_prompt_key(self) -> str:
        return "crossllm"