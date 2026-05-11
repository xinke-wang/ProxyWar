"""
OpenRouter-based code generation classes.

This module implements code generators that use OpenRouter platform
to access various LLMs for agent code generation.
"""

import os
from typing import Dict, Any, Optional, List
from openai import OpenAI

from .base import BaseCoder
from ..registry import CODER_REGISTRY


class OpenRouterCoder(BaseCoder):
    """
    Base class for OpenRouter-based code generators.
    
    This class handles common OpenRouter API interactions and can be
    extended for specific models with custom prompting strategies.
    """
    
    def __init__(self, 
                 model_name: str,
                 openrouter_model_id: str,
                 model_category: str = BaseCoder.GENERAL,
                 site_url: Optional[str] = None,
                 site_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize OpenRouter coder.
        
        Args:
            model_name: Human-readable name for the model
            openrouter_model_id: OpenRouter model identifier (e.g., "minimax/minimax-m1")
            model_category: Category of the model (general, reasoning, code)
            site_url: Optional site URL for headers
            site_name: Optional site name for headers
            **kwargs: Additional configuration
        """
        super().__init__(model_name, model_category, **kwargs)
        
        self.openrouter_model_id = openrouter_model_id
        self.site_url = site_url
        self.site_name = site_name
        
        # Initialize OpenAI client with OpenRouter configuration
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    
    def generate_agent_code(self, 
                          prompt: str) -> str:
        """
        Generate agent code using OpenRouter API.
        
        Args:
            prompt: The complete prompt string to send to the model
            
        Returns:
            Generated Python code as a string
        """
        
        # Prepare headers
        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra_headers["X-Title"] = self.site_name
        
        try:
            completion = self.client.chat.completions.create(
                extra_headers=extra_headers,
                extra_body={},
                model=self.openrouter_model_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                **self.config
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate code with {self.model_name}: {str(e)}")
    
    def _generate_with_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """
        Generate code using conversation history for multi-round chat.
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            Generated code as string
        """
        # Prepare headers
        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra_headers["X-Title"] = self.site_name
        
        try:
            # Convert conversation history to proper format
            messages = []
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            completion = self.client.chat.completions.create(
                extra_headers=extra_headers,
                extra_body={},
                model=self.openrouter_model_id,
                messages=messages,
                **self.config
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate revised code with {self.model_name}: {str(e)}")


@CODER_REGISTRY.register('minimax_m1')
class MiniMaxM1Coder(OpenRouterCoder):
    """
    MiniMax-M1 specific code generator.
    
    This class implements code generation using the MiniMax-M1 model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize MiniMax-M1 coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="MiniMax-M1",
            openrouter_model_id="minimax/minimax-m1",
            model_category=BaseCoder.REASONING,
            **kwargs
        )


@CODER_REGISTRY.register('phi4')
class Phi4Coder(OpenRouterCoder):
    """
    Microsoft Phi-4 specific code generator.
    
    This class implements code generation using the Microsoft Phi-4 model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize Phi-4 coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="Microsoft Phi-4",
            openrouter_model_id="microsoft/phi-4",
            model_category=BaseCoder.GENERAL,
            **kwargs
        )
    
 
@CODER_REGISTRY.register('gpt4_1_mini')
class GPT41MiniCoder(OpenRouterCoder):
    """
    OpenAI GPT-4.1 Mini specific code generator.
    
    This class implements code generation using the GPT-4.1 Mini model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize GPT-4.1 Mini coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="GPT-4.1 Mini",
            openrouter_model_id="openai/gpt-4.1-mini-2025-04-14",
            model_category=BaseCoder.GENERAL,
            **kwargs
        )
    

@CODER_REGISTRY.register('o3_mini')
class O3MiniCoder(OpenRouterCoder):
    """
    OpenAI O3-Mini specific code generator.
    
    This class implements code generation using the O3-Mini model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize O3-Mini coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="OpenAI O3-Mini",
            openrouter_model_id="openai/o3-mini-2025-01-31",
            model_category=BaseCoder.REASONING,
            **kwargs
        )
    

@CODER_REGISTRY.register('qwen_2_5_coder_32b')
class Qwen25Coder32BCoder(OpenRouterCoder):
    """
    Qwen 2.5 Coder 32B specific code generator.
    
    This class implements code generation using the Qwen 2.5 Coder 32B model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize Qwen 2.5 Coder 32B coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="Qwen 2.5 Coder 32B",
            openrouter_model_id="qwen/qwen-2.5-coder-32b-instruct",
            model_category=BaseCoder.CODE,
            **kwargs
        )
    

@CODER_REGISTRY.register('codestral_2501')
class Codestral2501Coder(OpenRouterCoder):
    """
    Mistral Codestral 2501 specific code generator.
    
    This class implements code generation using the Mistral Codestral 2501 model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize Mistral Codestral 2501 coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="Mistral Codestral 2501",
            openrouter_model_id="mistralai/codestral-2501",
            model_category=BaseCoder.CODE,
            **kwargs
        )
    
 
@CODER_REGISTRY.register('gemini_2_5_flash')
class Gemini25FlashCoder(OpenRouterCoder):
    """
    Google Gemini 2.5 Flash specific code generator.
    
    This class implements code generation using the Google Gemini 2.5 Flash model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize Gemini 2.5 Flash coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="Google Gemini 2.5 Flash",
            openrouter_model_id="google/gemini-2.5-flash",
            model_category=BaseCoder.GENERAL,
            **kwargs
        )
    
    

@CODER_REGISTRY.register('claude_sonnet_4')
class ClaudeSonnet4Coder(OpenRouterCoder):
    """
    Anthropic Claude Sonnet 4 specific code generator.
    
    This class implements code generation using the Anthropic Claude Sonnet 4 model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize Claude Sonnet 4 coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="Anthropic Claude Sonnet 4",
            openrouter_model_id="anthropic/claude-4-sonnet-20250522",
            model_category=BaseCoder.REASONING,
            **kwargs
        )
    
 
@CODER_REGISTRY.register('gpt4_1')
class GPT41Coder(OpenRouterCoder):
    """
    OpenAI GPT-4.1 specific code generator.
    
    This class implements code generation using the OpenAI GPT-4.1 model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize GPT-4.1 coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="OpenAI GPT-4.1",
            openrouter_model_id="openai/gpt-4.1-2025-04-14",
            model_category=BaseCoder.GENERAL,
            **kwargs
        )
    

@CODER_REGISTRY.register('llama_4_maverick')
class Llama4MaverickCoder(OpenRouterCoder):
    """
    Meta Llama 4 Maverick specific code generator.
    
    This class implements code generation using the Meta Llama 4 Maverick model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize Llama 4 Maverick coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="Meta Llama 4 Maverick",
            openrouter_model_id="meta-llama/llama-4-maverick-17b-128e-instruct",
            model_category=BaseCoder.GENERAL,
            **kwargs
        )
    
 
@CODER_REGISTRY.register('qwen3_235b_a22b')
class Qwen3235BA22BCoder(OpenRouterCoder):
    """
    Qwen Qwen3-235B-A22B specific code generator.
    
    This class implements code generation using the Qwen Qwen3-235B-A22B model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize Qwen3-235B-A22B coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="Qwen Qwen3-235B-A22B",
            openrouter_model_id="qwen/qwen3-235b-a22b-04-28",
            model_category=BaseCoder.REASONING,
            **kwargs
        )
    
 
@CODER_REGISTRY.register('deepseek_chat_v3')
class DeepSeekChatV3Coder(OpenRouterCoder):
    """
    DeepSeek Chat V3 specific code generator.
    
    This class implements code generation using the DeepSeek Chat V3 model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize DeepSeek Chat V3 coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="DeepSeek Chat V3",
            openrouter_model_id="deepseek/deepseek-chat-v3-0324",
            model_category=BaseCoder.GENERAL,
            **kwargs
        )
    
 
@CODER_REGISTRY.register('deepseek_r1')
class DeepSeekR1Coder(OpenRouterCoder):
    """
    DeepSeek R1 specific code generator.
    
    This class implements code generation using the DeepSeek R1 model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize DeepSeek R1 coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="DeepSeek R1",
            openrouter_model_id="deepseek/deepseek-r1-0528",
            model_category=BaseCoder.REASONING,
            **kwargs
        )
    
 
@CODER_REGISTRY.register('mistral_magistral_small')
class MistralMagistralSmallCoder(OpenRouterCoder):
    """
    Mistral Magistral Small specific code generator.
    
    This class implements code generation using the Mistral Magistral Small model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize Mistral Magistral Small coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="Mistral Magistral Small",
            openrouter_model_id="mistralai/magistral-small-2506",
            model_category=BaseCoder.REASONING,
            **kwargs
        )
    
 
@CODER_REGISTRY.register('qwen_2_5_72b_instruct')
class Qwen25_72BInstructCoder(OpenRouterCoder):
    """
    Qwen 2.5 72B Instruct specific code generator.
    
    This class implements code generation using the Qwen 2.5 72B Instruct model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize Qwen 2.5 72B Instruct coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="Qwen 2.5 72B Instruct",
            openrouter_model_id="qwen/qwen-2.5-72b-instruct",
            model_category=BaseCoder.GENERAL,
            **kwargs
        )
    
 
@CODER_REGISTRY.register('openai_codex_mini')
class OpenAICodexMiniCoder(OpenRouterCoder):
    """
    OpenAI Codex Mini specific code generator.
    
    This class implements code generation using the OpenAI Codex Mini model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize OpenAI Codex Mini coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="OpenAI Codex Mini",
            openrouter_model_id="openai/codex-mini",
            model_category=BaseCoder.CODE,
            **kwargs
        )
    

@CODER_REGISTRY.register('inception_mercury_coder')
class InceptionMercuryCoderCoder(OpenRouterCoder):
    """
    Inception Mercury Coder specific code generator.
    
    This class implements code generation using the Inception Mercury Coder model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize Inception Mercury Coder coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="Inception Mercury Coder",
            openrouter_model_id="inception/mercury-coder",
            model_category=BaseCoder.CODE,
            **kwargs
        )
    
 
@CODER_REGISTRY.register('claude_3_5_sonnet')
class Claude35SonnetCoder(OpenRouterCoder):
    """
    Anthropic Claude 3.5 Sonnet specific code generator.
    
    This class implements code generation using the Anthropic Claude 3.5 Sonnet model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize Claude 3.5 Sonnet coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="Anthropic Claude 3.5 Sonnet",
            openrouter_model_id="anthropic/claude-3.5-sonnet",
            model_category=BaseCoder.GENERAL,
            **kwargs
        )
    
 
@CODER_REGISTRY.register('gemini_2_0_flash_001')
class Gemini20Flash001Coder(OpenRouterCoder):
    """
    Google Gemini 2.0 Flash 001 specific code generator.
    
    This class implements code generation using the Google Gemini 2.0 Flash 001 model
    through OpenRouter platform.
    """
    
    def __init__(self, 
                 **kwargs):
        """
        Initialize Gemini 2.0 Flash 001 coder.
        
        Args:
            **kwargs: Additional model configuration (temperature, max_tokens, etc.)
        """
        super().__init__(
            model_name="Google Gemini 2.0 Flash 001",
            openrouter_model_id="google/gemini-2.0-flash-001",
            model_category=BaseCoder.GENERAL,
            **kwargs
        )
    
 