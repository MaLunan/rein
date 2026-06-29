"""Provider 子系统:模型接入层。

- Provider:统一接口(协议)
- MockProvider:测试用(核心,不联网)
- LiteLLMProvider:接真实模型(延迟 import litellm)
- FallbackProvider:主/备自动切换(M1)
"""

from rein.providers.base import Provider
from rein.providers.fallback import FallbackProvider
from rein.providers.litellm import LiteLLMProvider
from rein.providers.mock import MockProvider

__all__ = ["Provider", "MockProvider", "LiteLLMProvider", "FallbackProvider"]
