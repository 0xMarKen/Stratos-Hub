"""
LLM Integration Module

Provides unified interface for interacting with multiple LLM providers
including OpenAI, Anthropic, Google, Cohere, and local models.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, AsyncIterator, Any
from decimal import Decimal

import aiohttp
import openai
import anthropic
import google.generativeai as genai
import cohere
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import tiktoken

from ..core.config import get_settings
from ..core.logging import get_logger
from ..core.metrics import MetricsCollector
from ..monitoring.usage_tracker import UsageTracker

settings = get_settings()
logger = get_logger(__name__)


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    CUSTOM = "custom"


class ModelCapability(Enum):
    TEXT_GENERATION = "text_generation"
    TEXT_COMPLETION = "text_completion"
    CHAT_COMPLETION = "chat_completion"
    CODE_GENERATION = "code_generation"
    FUNCTION_CALLING = "function_calling"
    EMBEDDINGS = "embeddings"
    FINE_TUNING = "fine_tuning"
    MULTIMODAL = "multimodal"


@dataclass
class LLMConfig:
    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 10000
    capabilities: List[ModelCapability] = field(default_factory=list)
    cost_per_1k_tokens: Decimal = Decimal('0.001')


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant", "function"
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


@dataclass
class LLMRequest:
    messages: List[ChatMessage]
    model: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    stream: bool = False
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    function_call: Optional[Dict[str, Any]] = None
    response_time: float = 0.0
    token_count: int = 0
    cost: Decimal = Decimal('0')
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingChunk:
    content: str
    finish_reason: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, int]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.metrics = MetricsCollector()
        self.usage_tracker = UsageTracker()
        
    @abstractmethod
    async def chat_completion(self, request: LLMRequest) -> LLMResponse:
        """Generate chat completion"""
        pass
    
    @abstractmethod
    async def stream_completion(self, request: LLMRequest) -> AsyncIterator[StreamingChunk]:
        """Generate streaming chat completion"""
        pass
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate text embeddings"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT models provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(config.model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    async def chat_completion(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    **({"name": msg.name} if msg.name else {}),
                    **({"function_call": msg.function_call} if msg.function_call else {})
                }
                for msg in request.messages
            ]
            
            # Prepare request parameters
            params = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or self.config.temperature,
                "top_p": request.top_p or self.config.top_p,
                "frequency_penalty": request.frequency_penalty or self.config.frequency_penalty,
                "presence_penalty": request.presence_penalty or self.config.presence_penalty,
                "stop": request.stop or self.config.stop_sequences or None,
                "stream": False
            }
            
            # Add function calling parameters if provided
            if request.functions:
                params["functions"] = request.functions
            if request.function_call:
                params["function_call"] = request.function_call
            
            # Make API call
            response = await self.client.chat.completions.create(**params)
            
            # Calculate metrics
            response_time = time.time() - start_time
            usage = response.usage.dict() if response.usage else {}
            token_count = usage.get('total_tokens', 0)
            cost = self.calculate_cost(usage)
            
            # Record metrics
            await self.record_metrics(request, response_time, token_count, cost, True)
            
            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""
            function_call = choice.message.function_call.dict() if choice.message.function_call else None
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=choice.finish_reason,
                function_call=function_call,
                response_time=response_time,
                token_count=token_count,
                cost=cost
            )
            
        except Exception as e:
            await self.record_metrics(request, time.time() - start_time, 0, Decimal('0'), False)
            logger.error(f"OpenAI API error: {e}")
            raise LLMError(f"OpenAI API error: {e}")
    
    async def stream_completion(self, request: LLMRequest) -> AsyncIterator[StreamingChunk]:
        try:
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            params = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or self.config.temperature,
                "stream": True
            }
            
            stream = await self.client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    yield StreamingChunk(
                        content=delta.content or "",
                        finish_reason=choice.finish_reason,
                        function_call=delta.function_call.dict() if delta.function_call else None
                    )
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise LLMError(f"OpenAI streaming error: {e}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            
            return [embedding.embedding for embedding in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI embeddings error: {e}")
            raise LLMError(f"OpenAI embeddings error: {e}")
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def calculate_cost(self, usage: Dict[str, int]) -> Decimal:
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        
        # OpenAI pricing (example rates)
        if "gpt-4" in self.config.model_name:
            prompt_cost = Decimal(str(prompt_tokens)) * Decimal('0.03') / 1000
            completion_cost = Decimal(str(completion_tokens)) * Decimal('0.06') / 1000
        elif "gpt-3.5-turbo" in self.config.model_name:
            prompt_cost = Decimal(str(prompt_tokens)) * Decimal('0.0015') / 1000
            completion_cost = Decimal(str(completion_tokens)) * Decimal('0.002') / 1000
        else:
            # Default pricing
            total_tokens = prompt_tokens + completion_tokens
            return Decimal(str(total_tokens)) * self.config.cost_per_1k_tokens / 1000
        
        return prompt_cost + completion_cost


class AnthropicProvider(LLMProvider):
    """Anthropic Claude models provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = anthropic.AsyncAnthropic(
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
    
    async def chat_completion(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()
        
        try:
            # Convert messages to Anthropic format
            system_message = ""
            messages = []
            
            for msg in request.messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            params = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or self.config.temperature,
                "top_p": request.top_p or self.config.top_p,
                "stop_sequences": request.stop or self.config.stop_sequences or None
            }
            
            if system_message:
                params["system"] = system_message
            
            response = await self.client.messages.create(**params)
            
            response_time = time.time() - start_time
            token_count = response.usage.input_tokens + response.usage.output_tokens
            cost = self.calculate_cost(response.usage)
            
            await self.record_metrics(request, response_time, token_count, cost, True)
            
            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": token_count
                },
                finish_reason=response.stop_reason,
                response_time=response_time,
                token_count=token_count,
                cost=cost
            )
            
        except Exception as e:
            await self.record_metrics(request, time.time() - start_time, 0, Decimal('0'), False)
            logger.error(f"Anthropic API error: {e}")
            raise LLMError(f"Anthropic API error: {e}")
    
    async def stream_completion(self, request: LLMRequest) -> AsyncIterator[StreamingChunk]:
        try:
            # Implementation for Claude streaming
            # Note: Anthropic streaming API implementation would go here
            pass
            
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise LLMError(f"Anthropic streaming error: {e}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Anthropic doesn't provide embeddings directly
        raise NotImplementedError("Anthropic doesn't provide embeddings API")
    
    def count_tokens(self, text: str) -> int:
        # Anthropic token counting (approximation)
        return len(text.split()) * 1.3  # Rough estimate
    
    def calculate_cost(self, usage) -> Decimal:
        # Claude pricing
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        
        input_cost = Decimal(str(input_tokens)) * Decimal('0.008') / 1000
        output_cost = Decimal(str(output_tokens)) * Decimal('0.024') / 1000
        
        return input_cost + output_cost


class HuggingFaceProvider(LLMProvider):
    """Hugging Face Transformers provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    async def load_model(self):
        """Load model and tokenizer"""
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    async def chat_completion(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()
        
        try:
            await self.load_model()
            
            # Format messages into a single prompt
            prompt = self.format_messages_for_model(request.messages)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=request.max_tokens or self.config.max_tokens,
                    temperature=request.temperature or self.config.temperature,
                    top_p=request.top_p or self.config.top_p,
                    top_k=request.top_k or self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response_text = self.tokenizer.decode(
                outputs[0][inputs.shape[-1]:], 
                skip_special_tokens=True
            )
            
            response_time = time.time() - start_time
            token_count = len(outputs[0])
            cost = Decimal('0')  # Local model, no cost
            
            await self.record_metrics(request, response_time, token_count, cost, True)
            
            return LLMResponse(
                content=response_text.strip(),
                model=self.config.model_name,
                usage={
                    "prompt_tokens": len(inputs[0]),
                    "completion_tokens": len(outputs[0]) - len(inputs[0]),
                    "total_tokens": len(outputs[0])
                },
                finish_reason="stop",
                response_time=response_time,
                token_count=token_count,
                cost=cost
            )
            
        except Exception as e:
            await self.record_metrics(request, time.time() - start_time, 0, Decimal('0'), False)
            logger.error(f"HuggingFace model error: {e}")
            raise LLMError(f"HuggingFace model error: {e}")
    
    async def stream_completion(self, request: LLMRequest) -> AsyncIterator[StreamingChunk]:
        try:
            await self.load_model()
            
            prompt = self.format_messages_for_model(request.messages)
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Setup streaming
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True
            )
            
            generation_kwargs = {
                "input_ids": inputs,
                "max_new_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or self.config.temperature,
                "top_p": request.top_p or self.config.top_p,
                "do_sample": True,
                "streamer": streamer,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            
            # Start generation in background thread
            import threading
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream tokens
            for token in streamer:
                yield StreamingChunk(content=token)
            
            thread.join()
            
        except Exception as e:
            logger.error(f"HuggingFace streaming error: {e}")
            raise LLMError(f"HuggingFace streaming error: {e}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Would need a separate embedding model
        raise NotImplementedError("Use a dedicated embedding model")
    
    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())  # Approximation
    
    def format_messages_for_model(self, messages: List[ChatMessage]) -> str:
        """Format chat messages for models that expect a single prompt"""
        formatted = ""
        for msg in messages:
            if msg.role == "system":
                formatted += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                formatted += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                formatted += f"Assistant: {msg.content}\n\n"
        
        formatted += "Assistant: "
        return formatted


class LLMIntegration:
    """Main LLM integration class that manages multiple providers"""
    
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.default_provider: Optional[str] = None
        self.metrics = MetricsCollector()
        self.usage_tracker = UsageTracker()
        
    def register_provider(self, name: str, provider: LLMProvider, is_default: bool = False):
        """Register a new LLM provider"""
        self.providers[name] = provider
        if is_default or self.default_provider is None:
            self.default_provider = name
        
        logger.info(f"Registered LLM provider: {name}")
    
    async def chat_completion(
        self, 
        request: LLMRequest, 
        provider_name: Optional[str] = None
    ) -> LLMResponse:
        """Generate chat completion using specified or default provider"""
        provider_name = provider_name or self.default_provider
        
        if not provider_name or provider_name not in self.providers:
            raise LLMError(f"Provider {provider_name} not found")
        
        provider = self.providers[provider_name]
        
        # Rate limiting check
        if not await self.check_rate_limits(provider_name, request):
            raise RateLimitError(f"Rate limit exceeded for provider {provider_name}")
        
        # Execute request
        response = await provider.chat_completion(request)
        
        # Track usage
        await self.usage_tracker.track_usage(
            provider_name,
            request.user_id,
            response.token_count,
            response.cost
        )
        
        return response
    
    async def stream_completion(
        self, 
        request: LLMRequest, 
        provider_name: Optional[str] = None
    ) -> AsyncIterator[StreamingChunk]:
        """Generate streaming completion"""
        provider_name = provider_name or self.default_provider
        
        if not provider_name or provider_name not in self.providers:
            raise LLMError(f"Provider {provider_name} not found")
        
        provider = self.providers[provider_name]
        
        if not await self.check_rate_limits(provider_name, request):
            raise RateLimitError(f"Rate limit exceeded for provider {provider_name}")
        
        async for chunk in provider.stream_completion(request):
            yield chunk
    
    async def generate_embeddings(
        self, 
        texts: List[str], 
        provider_name: Optional[str] = None
    ) -> List[List[float]]:
        """Generate text embeddings"""
        provider_name = provider_name or self.default_provider
        
        if not provider_name or provider_name not in self.providers:
            raise LLMError(f"Provider {provider_name} not found")
        
        provider = self.providers[provider_name]
        return await provider.generate_embeddings(texts)
    
    def count_tokens(self, text: str, provider_name: Optional[str] = None) -> int:
        """Count tokens for given text"""
        provider_name = provider_name or self.default_provider
        
        if not provider_name or provider_name not in self.providers:
            raise LLMError(f"Provider {provider_name} not found")
        
        provider = self.providers[provider_name]
        return provider.count_tokens(text)
    
    async def check_rate_limits(self, provider_name: str, request: LLMRequest) -> bool:
        """Check if request is within rate limits"""
        # Implementation would check current usage against limits
        return True  # Simplified for now
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models per provider"""
        models = {}
        for name, provider in self.providers.items():
            # This would be implemented based on each provider's model list
            models[name] = [provider.config.model_name]
        return models
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all providers"""
        health = {}
        for name, provider in self.providers.items():
            try:
                # Simple health check with minimal request
                test_request = LLMRequest(
                    messages=[ChatMessage(role="user", content="test")],
                    model=provider.config.model_name,
                    max_tokens=1
                )
                await provider.chat_completion(test_request)
                health[name] = True
            except Exception as e:
                logger.warning(f"Health check failed for provider {name}: {e}")
                health[name] = False
        
        return health


# Error classes
class LLMError(Exception):
    """Base LLM error"""
    pass


class RateLimitError(LLMError):
    """Rate limit exceeded error"""
    pass


class ModelNotFoundError(LLMError):
    """Model not found error"""
    pass


class TokenLimitError(LLMError):
    """Token limit exceeded error"""
    pass


# Helper function to record metrics
async def record_metrics(
    self, 
    request: LLMRequest, 
    response_time: float, 
    token_count: int, 
    cost: Decimal, 
    success: bool
):
    """Record metrics for LLM requests"""
    self.metrics.record_llm_request({
        "provider": self.config.provider.value,
        "model": request.model,
        "response_time": response_time,
        "token_count": token_count,
        "cost": float(cost),
        "success": success,
        "timestamp": time.time()
    })


# Add the record_metrics method to LLMProvider
LLMProvider.record_metrics = record_metrics 