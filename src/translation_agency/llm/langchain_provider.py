"""LangChain integration for enhanced LLM management and interactions."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Optional, Dict, Any
import os

class LangChainProvider:
    """LangChain-based LLM provider for more robust interactions."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", temperature: float = 0.1):
        """
        Initialize LangChain OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: OpenAI model to use
            temperature: Temperature for generation
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=model,
            temperature=temperature
        )

    def create_translation_chain(self, prompt_template: str) -> Any:
        """
        Create a LangChain chain for translation tasks.

        Args:
            prompt_template: Prompt template string

        Returns:
            Runnable chain for translation
        """
        prompt = PromptTemplate.from_template(prompt_template)

        chain = (
            {"input_text": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def translate(self, text: str, prompt_template: str, **kwargs) -> str:
        """
        Translate text using LangChain.

        Args:
            text: Text to translate
            prompt_template: Translation prompt template
            **kwargs: Additional template variables

        Returns:
            Translated text
        """
        chain = self.create_translation_chain(prompt_template)
        return chain.invoke({"input_text": text, **kwargs})

    def validate(self, text: str, prompt_template: str, **kwargs) -> str:
        """
        Validate text using LangChain.

        Args:
            text: Text to validate
            prompt_template: Validation prompt template
            **kwargs: Additional template variables

        Returns:
            Validation result
        """
        chain = self.create_translation_chain(prompt_template)
        return chain.invoke({"input_text": text, **kwargs})

class LangChainManager:
    """Manager for LangChain providers with fallback support."""

    def __init__(self, primary_provider: str = "textgenhub", fallback_provider: str = "langchain"):
        """
        Initialize LangChain manager with primary and fallback providers.

        Args:
            primary_provider: Primary LLM provider ("textgenhub" or "langchain")
            fallback_provider: Fallback provider
        """
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.providers = {}

        # Initialize providers
        if primary_provider == "langchain" or fallback_provider == "langchain":
            try:
                self.providers["langchain"] = LangChainProvider()
            except Exception as e:
                print(f"Warning: Could not initialize LangChain provider: {e}")

    def get_provider(self, provider_name: str):
        """Get a provider by name."""
        return self.providers.get(provider_name)

    def translate_with_fallback(self, text: str, prompt_template: str, **kwargs) -> str:
        """
        Translate with fallback support.

        Args:
            text: Text to translate
            prompt_template: Translation prompt template
            **kwargs: Additional arguments

        Returns:
            Translated text
        """
        # Try primary provider first
        try:
            if self.primary_provider == "langchain":
                provider = self.providers.get("langchain")
                if provider:
                    return provider.translate(text, prompt_template, **kwargs)
            # For textgenhub, we'd need to integrate it here
            # For now, fall back to langchain if available

        except Exception as e:
            print(f"Primary provider failed: {e}")

        # Try fallback
        try:
            if self.fallback_provider == "langchain":
                provider = self.providers.get("langchain")
                if provider:
                    return provider.translate(text, prompt_template, **kwargs)
        except Exception as e:
            print(f"Fallback provider also failed: {e}")

        raise RuntimeError("All LLM providers failed")

# Global instance for easy access
langchain_manager = LangChainManager()