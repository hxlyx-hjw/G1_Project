"""Non-invasive DeepSeek tool extensions for the G1 robot project.

The chat subclass is intentionally not imported here. Import
``g1_deepseek_ext.chat`` only after the G1 source directory is on ``sys.path``.
"""

from .adapters import DeepSeekIntegration, IntegrationConfig

__all__ = [
    "DeepSeekIntegration",
    "IntegrationConfig",
]
