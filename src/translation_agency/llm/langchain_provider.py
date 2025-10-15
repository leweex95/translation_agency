from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from textgenhub import ChatGPT
from typing import Optional, Dict, Any, List
import asyncio


class TextGenHubLLM(LLM):
    """LangChain-compatible wrapper for textgenhub.ChatGPT."""

    headless: bool = True
    remove_cache: bool = True
    debug: bool = False

    def __init__(self, headless: bool = True, remove_cache: bool = True, debug: bool = False):
        """Initialize with textgenhub.ChatGPT configuration."""
        super().__init__()
        self.headless = headless
        self.remove_cache = remove_cache
        self.debug = debug
        # Use object.__setattr__ to bypass Pydantic validation for internal state
        object.__setattr__(self, '_provider', None)

    @property
    def provider(self):
        """Lazy initialization of textgenhub provider."""
        if self._provider is None:
            self._provider = ChatGPT(
                headless=self.headless,
                remove_cache=self.remove_cache
            )
        return self._provider

    @property
    def model_name(self) -> str:
        """Return model name for LangChain compatibility."""
        return "textgenhub-chatgpt"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        """Execute LLM call through textgenhub.ChatGPT."""
        try:
            result = self.provider.chat(prompt)

            # Handle stop sequences if provided
            if stop:
                for stop_seq in stop:
                    if stop_seq in result:
                        result = result.split(stop_seq)[0]
                        break

            return result
        except Exception as e:
            raise RuntimeError(f"TextGenHub LLM call failed: {e}")

    @property
    def _llm_type(self) -> str:
        """Return LLM type identifier."""
        return "textgenhub-chatgpt"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model_name": self.model_name,
            "headless": self.headless,
            "remove_cache": self.remove_cache
        }


class TextGenHubLangChainProvider:
    """LangChain provider using textgenhub.ChatGPT exclusively."""

    def __init__(self, headless: bool = True, remove_cache: bool = True, debug: bool = False):
        """
        Initialize with textgenhub.ChatGPT configuration.

        Args:
            headless: Run browser in headless mode
            remove_cache: Remove cache between calls
            debug: Enable debug mode
        """
        self.llm = TextGenHubLLM(
            headless=headless,
            remove_cache=remove_cache,
            debug=debug
        )

    def create_step_chain(self, prompt_template: str, step_name: str):
        """Create a single validation step chain."""
        prompt = PromptTemplate.from_template(prompt_template)

        def log_step(inputs):
            print(f"[LangChain] Executing {step_name}")
            return inputs

        return (
            RunnableLambda(log_step)
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def create_translation_pipeline_chain(self, step_configs: List[Dict[str, Any]]):
        """
        Create a complete chained translation pipeline using textgenhub.

        Args:
            step_configs: List of dicts with 'prompt_template', 'step_name', and 'needs_original'

        Returns:
            Complete chained pipeline
        """
        chains = []

        for config in step_configs:
            chain = self.create_step_chain(
                config['prompt_template'],
                config['step_name']
            )
            chains.append((chain, config.get('needs_original', False)))

        def execute_chain(inputs):
            """Execute the complete chain with proper context passing."""
            original_text = inputs.get('original_text', '')
            current_text = inputs.get('text', '')

            for chain, needs_original in chains:
                # Prepare inputs for this step
                step_inputs = {'text': current_text}
                if needs_original:
                    step_inputs['original_text'] = original_text

                # Add any additional template variables
                for key, value in inputs.items():
                    if key not in step_inputs:
                        step_inputs[key] = value

                # Execute step
                current_text = chain.invoke(step_inputs)

            return current_text

        return RunnableLambda(execute_chain)

    def run_parallel_validation(self, text: str, original_text: str,
                              validation_configs: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Run multiple validation steps in parallel using textgenhub.

        Args:
            text: Text to validate
            original_text: Original source text
            validation_configs: List of validation step configurations

        Returns:
            Dictionary mapping step names to results
        """
        async def run_async_validation():
            tasks = []
            for config in validation_configs:
                chain = self.create_step_chain(
                    config['prompt_template'],
                    config['step_name']
                )

                inputs = {'text': text}
                if config.get('needs_original', False):
                    inputs['original_text'] = original_text

                # Add other template variables
                for key, value in config.items():
                    if key not in ['prompt_template', 'step_name', 'needs_original']:
                        inputs[key] = value

                # Create async task
                task = asyncio.create_task(chain.ainvoke(inputs))
                tasks.append((config['step_name'], task))

            results = {}
            for step_name, task in tasks:
                try:
                    results[step_name] = await task
                except Exception as e:
                    print(f"[Error] Step {step_name} failed: {e}")
                    results[step_name] = text  # Fallback to original

            return results

        # Run in asyncio context
        return asyncio.run(run_async_validation())

    def chat(self, prompt: str) -> str:
        """
        Direct chat method for compatibility.

        Args:
            prompt: Text prompt to process

        Returns:
            LLM response via textgenhub
        """
        return self.llm._call(prompt)


class EnhancedTextGenHubProvider(TextGenHubLangChainProvider):
    """Enhanced provider with fallback capabilities."""

    def __init__(self, fallback_provider=None, **kwargs):
        """
        Initialize with optional fallback.

        Args:
            fallback_provider: Fallback textgenhub instance
            **kwargs: Arguments for textgenhub configuration
        """
        super().__init__(**kwargs)
        self.fallback_provider = fallback_provider

    def chat_with_fallback(self, prompt: str) -> str:
        """
        Execute chat with fallback support.

        Args:
            prompt: Text prompt to process

        Returns:
            LLM response
        """
        try:
            return self.chat(prompt)
        except Exception as e:
            print(f"[Warning] Primary provider failed ({e}), trying fallback")
            if self.fallback_provider:
                return self.fallback_provider.chat(prompt)
            else:
                raise RuntimeError("Both primary and fallback providers failed") from e


def create_translation_pipeline_config(prompts_dict: Dict[str, str],
                                     target_language: str,
                                     style: str) -> List[Dict[str, Any]]:
    """
    Create pipeline configuration from your existing prompts.

    Args:
        prompts_dict: Dictionary of prompt templates
        target_language: Target language for translation
        style: Translation style

    Returns:
        List of step configurations for chaining
    """
    step_configs = [
        {
            'prompt_template': prompts_dict['translation'],
            'step_name': 'initial_translation',
            'target_language': target_language,
            'style': style,
            'needs_original': False
        },
        {
            'prompt_template': prompts_dict['grammar'],
            'step_name': 'grammar_validation',
            'style': style,
            'needs_original': False
        },
        {
            'prompt_template': prompts_dict['style'],
            'step_name': 'style_validation',
            'style': style,
            'needs_original': False
        },
        {
            'prompt_template': prompts_dict['accuracy'],
            'step_name': 'accuracy_validation',
            'style': style,
            'needs_original': True
        },
        {
            'prompt_template': prompts_dict['hallucination'],
            'step_name': 'hallucination_validation',
            'style': style,
            'needs_original': True
        },
        {
            'prompt_template': prompts_dict['consistency'],
            'step_name': 'consistency_validation',
            'style': style,
            'needs_original': False
        },
        {
            'prompt_template': prompts_dict['crossllm'],
            'step_name': 'crossllm_validation',
            'style': style,
            'needs_original': False
        }
    ]

    return step_configs

