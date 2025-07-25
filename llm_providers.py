"""
Flexible LLM provider system for healthcare facility extraction
Supports OpenRouter, local models (Ollama), and other providers
"""

import json
import logging
import os
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import aiohttp
import requests


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL_API = "local_api"


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 60


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    async def generate(self, prompt: str, schema: Optional[Dict] = None) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter API provider - access to many open source and commercial models"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://openrouter.ai/api/v1"
        self.api_key = config.api_key or os.getenv('OPENROUTER_API_KEY')
        
        # Popular models available on OpenRouter
        self.recommended_models = {
            'fast': 'meta-llama/llama-3.1-8b-instruct:free',  # Free tier
            'balanced': 'meta-llama/llama-3.1-70b-instruct',  # Good balance
            'powerful': 'anthropic/claude-3.5-sonnet',        # High accuracy
            'coding': 'deepseek/deepseek-coder-33b-instruct', # Good for structured data
            'cheap': 'google/gemma-2-9b-it:free',             # Free alternative
        }
    
    async def generate(self, prompt: str, schema: Optional[Dict] = None) -> str:
        """Generate response using OpenRouter API"""
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/healthcare-scraper",  # Optional
            "X-Title": "Healthcare Facility Scraper"  # Optional
        }
        
        # Prepare the request
        messages = [{"role": "user", "content": prompt}]
        
        # Add schema instruction if provided
        if schema:
            schema_prompt = f"\n\nPlease respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
            messages[0]["content"] += schema_prompt
        
        data = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                async with session.post(f"{self.base_url}/chat/completions", headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        raise Exception(f"OpenRouter API error {response.status}: {error_text}")
        
        except Exception as e:
            self.logger.error(f"OpenRouter generation failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if OpenRouter is available"""
        return bool(self.api_key)
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about available models"""
        return {
            "current_model": self.config.model,
            "recommended_models": self.recommended_models,
            "provider": "OpenRouter",
            "cost": "Pay-per-use, many free models available"
        }


class OllamaProvider(BaseLLMProvider):
    """Local Ollama provider for running models locally"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        
        # Popular models for Ollama
        self.recommended_models = {
            'fast': 'llama3.2:3b',           # Fast, good for basic tasks
            'balanced': 'llama3.1:8b',       # Good balance of speed/quality
            'powerful': 'llama3.1:70b',      # High quality (requires lots of RAM)
            'coding': 'codellama:13b',       # Good for structured data
            'small': 'phi3:mini',            # Very fast, minimal resources
        }
    
    async def generate(self, prompt: str, schema: Optional[Dict] = None) -> str:
        """Generate response using local Ollama"""
        
        # Add schema instruction if provided
        if schema:
            prompt += f"\n\nPlease respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        
        data = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                async with session.post(f"{self.base_url}/api/generate", json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['response']
                    else:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error {response.status}: {error_text}")
        
        except Exception as e:
            self.logger.error(f"Ollama generation failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of locally available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except:
            pass
        return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model to local Ollama instance"""
        try:
            self.logger.info(f"Pulling model {model_name}...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300  # 5 minutes for model download
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to pull model {model_name}: {e}")
            return False


class LocalAPIProvider(BaseLLMProvider):
    """Generic local API provider (llama.cpp, text-generation-webui, etc.)"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:8000"
    
    async def generate(self, prompt: str, schema: Optional[Dict] = None) -> str:
        """Generate response using local API"""
        
        if schema:
            prompt += f"\n\nPlease respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        
        # Try different API formats
        endpoints_to_try = [
            # OpenAI-compatible format
            {
                "url": f"{self.base_url}/v1/chat/completions",
                "data": {
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                },
                "response_path": ["choices", 0, "message", "content"]
            },
            # Simple completion format
            {
                "url": f"{self.base_url}/api/v1/generate",
                "data": {
                    "prompt": prompt,
                    "max_new_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature
                },
                "response_path": ["results", 0, "text"]
            },
            # Text-generation-webui format
            {
                "url": f"{self.base_url}/api/v1/completions",
                "data": {
                    "prompt": prompt,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature
                },
                "response_path": ["choices", 0, "text"]
            }
        ]
        
        for endpoint in endpoints_to_try:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                    async with session.post(endpoint["url"], json=endpoint["data"]) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            # Navigate to response text using path
                            text = result
                            for key in endpoint["response_path"]:
                                text = text[key]
                            
                            return text
            except Exception as e:
                self.logger.debug(f"Endpoint {endpoint['url']} failed: {e}")
                continue
        
        raise Exception("All local API endpoints failed")
    
    def is_available(self) -> bool:
        """Check if local API is available"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            try:
                # Try alternative health check
                response = requests.get(f"{self.base_url}/v1/models", timeout=5)
                return response.status_code == 200
            except:
                return False


class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(config: LLMConfig) -> BaseLLMProvider:
        """Create appropriate provider based on config"""
        
        if config.provider == LLMProvider.OPENROUTER:
            return OpenRouterProvider(config)
        elif config.provider == LLMProvider.OLLAMA:
            return OllamaProvider(config)
        elif config.provider == LLMProvider.LOCAL_API:
            return LocalAPIProvider(config)
        elif config.provider == LLMProvider.OPENAI:
            # Fallback to OpenAI if needed
            from llm_extractors import LLMFacilityExtractor
            # Would need to adapt existing OpenAI code
            raise NotImplementedError("OpenAI provider - use original llm_extractors.py")
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    @staticmethod
    def auto_detect_provider() -> Optional[LLMConfig]:
        """Auto-detect available LLM provider"""
        
        # Check for OpenRouter API key
        if os.getenv('OPENROUTER_API_KEY'):
            return LLMConfig(
                provider=LLMProvider.OPENROUTER,
                model='meta-llama/llama-3.1-8b-instruct:free',  # Free model
                api_key=os.getenv('OPENROUTER_API_KEY')
            )
        
        # Check for local Ollama
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                if models:
                    return LLMConfig(
                        provider=LLMProvider.OLLAMA,
                        model=models[0]['name'],  # Use first available model
                        base_url="http://localhost:11434"
                    )
        except:
            pass
        
        # Check for local API servers
        local_urls = [
            "http://localhost:8000",
            "http://localhost:5000", 
            "http://localhost:7860"
        ]
        
        for url in local_urls:
            try:
                response = requests.get(f"{url}/v1/models", timeout=2)
                if response.status_code == 200:
                    return LLMConfig(
                        provider=LLMProvider.LOCAL_API,
                        model="local-model",
                        base_url=url
                    )
            except:
                continue
        
        # Check for OpenAI as fallback
        if os.getenv('OPENAI_API_KEY'):
            return LLMConfig(
                provider=LLMProvider.OPENAI,
                model='gpt-4o-mini',
                api_key=os.getenv('OPENAI_API_KEY')
            )
        
        return None


class MultiProviderLLM:
    """LLM client that can use multiple providers with fallback"""
    
    def __init__(self, primary_config: LLMConfig, fallback_configs: Optional[List[LLMConfig]] = None):
        self.primary_provider = LLMProviderFactory.create_provider(primary_config)
        self.fallback_providers = []
        
        if fallback_configs:
            for config in fallback_configs:
                try:
                    provider = LLMProviderFactory.create_provider(config)
                    if provider.is_available():
                        self.fallback_providers.append(provider)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Failed to create fallback provider: {e}")
        
        self.logger = logging.getLogger(__name__)
    
    async def generate(self, prompt: str, schema: Optional[Dict] = None) -> str:
        """Generate response with automatic fallback"""
        
        # Try primary provider first
        if self.primary_provider.is_available():
            try:
                return await self.primary_provider.generate(prompt, schema)
            except Exception as e:
                self.logger.warning(f"Primary provider failed: {e}")
        
        # Try fallback providers
        for i, provider in enumerate(self.fallback_providers):
            try:
                self.logger.info(f"Trying fallback provider {i+1}")
                return await provider.generate(prompt, schema)
            except Exception as e:
                self.logger.warning(f"Fallback provider {i+1} failed: {e}")
        
        raise Exception("All LLM providers failed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        return {
            "primary": {
                "available": self.primary_provider.is_available(),
                "config": self.primary_provider.config
            },
            "fallbacks": [
                {
                    "available": provider.is_available(),
                    "config": provider.config
                }
                for provider in self.fallback_providers
            ]
        }


# Convenience functions for easy setup
def setup_openrouter(model: str = "meta-llama/llama-3.1-8b-instruct:free", api_key: str = None) -> LLMConfig:
    """Setup OpenRouter configuration"""
    return LLMConfig(
        provider=LLMProvider.OPENROUTER,
        model=model,
        api_key=api_key or os.getenv('OPENROUTER_API_KEY')
    )


def setup_ollama(model: str = "llama3.1:8b", base_url: str = "http://localhost:11434") -> LLMConfig:
    """Setup Ollama configuration"""
    return LLMConfig(
        provider=LLMProvider.OLLAMA,
        model=model,
        base_url=base_url
    )


def setup_local_api(model: str = "local-model", base_url: str = "http://localhost:8000") -> LLMConfig:
    """Setup local API configuration"""
    return LLMConfig(
        provider=LLMProvider.LOCAL_API,
        model=model,
        base_url=base_url
    )


async def test_provider(config: LLMConfig) -> bool:
    """Test if a provider configuration works"""
    try:
        provider = LLMProviderFactory.create_provider(config)
        if not provider.is_available():
            return False
        
        response = await provider.generate("Hello, respond with just 'OK'")
        return "ok" in response.lower()
    
    except Exception as e:
        logging.getLogger(__name__).error(f"Provider test failed: {e}")
        return False

