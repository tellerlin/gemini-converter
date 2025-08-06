"""
Enhanced configuration management for Gemini Claude Adapter - Streamlined Version
"""

import os
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AppConfig(BaseSettings):
    """Main application configuration - Streamlined for current features"""
    
    # =============================================
    # Service Configuration
    # =============================================
    SERVICE_ENVIRONMENT: Environment = Field(Environment.DEVELOPMENT, description="Runtime environment")
    SERVICE_HOST: str = Field("0.0.0.0", description="Service host")
    SERVICE_PORT: int = Field(8100, description="Service port")
    SERVICE_LOG_LEVEL: LogLevel = Field(LogLevel.INFO, description="Log level")
    SERVICE_CORS_ORIGINS: Union[str, List[str]] = Field(default="*", description="CORS allowed origins")
    
    # =============================================
    # Gemini API Configuration - [REQUIRED]
    # =============================================
    GEMINI_API_KEYS: Union[str, List[str]] = Field(default="", description="Comma-separated Gemini API keys")
    GEMINI_COOLING_PERIOD: int = Field(300, description="Cooling period in seconds for a failed key", ge=60)
    GEMINI_REQUEST_TIMEOUT: int = Field(120, description="Request timeout in seconds for Gemini API calls", ge=10)
    GEMINI_MAX_RETRIES: int = Field(3, description="Maximum failure count before a key is marked as permanently failed", ge=1)
    
    # =============================================
    # Security Configuration - [REQUIRED]
    # =============================================
    SECURITY_ADAPTER_API_KEYS: Union[str, List[str]] = Field(default="", description="Comma-separated client API keys for accessing the adapter")
    SECURITY_ADMIN_API_KEYS: Union[str, List[str]] = Field(default="", description="Comma-separated admin API keys for management endpoints")
    
    # =============================================
    # Cache Configuration
    # =============================================
    CACHE_ENABLED: bool = Field(True, description="Enable response caching for non-streaming requests")
    CACHE_MAX_SIZE: int = Field(1000, description="Maximum number of items in the cache")
    CACHE_TTL: int = Field(300, description="Cache Time-To-Live in seconds")
    CACHE_KEY_PREFIX: str = Field("gemini_adapter", description="Prefix for cache keys")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )
    
    @field_validator('SECURITY_ADAPTER_API_KEYS', 'SECURITY_ADMIN_API_KEYS', 'GEMINI_API_KEYS', mode='before')
    @classmethod
    def validate_str_to_list(cls, v):
        """Validate and clean comma-separated strings into lists."""
        if v is None:
            return []
        if isinstance(v, str):
            return [key.strip() for key in v.split(',') if key.strip()]
        if isinstance(v, list):
            return [str(key).strip() for key in v if str(key).strip()]
        return v
    
    @field_validator('SERVICE_CORS_ORIGINS', mode='before')
    @classmethod
    def validate_cors_origins(cls, v):
        """Validate CORS origins from a comma-separated string."""
        if v is None or v == "":
            return ["*"]
        if isinstance(v, str):
            if v.strip() == "*":
                return ["*"]
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        if isinstance(v, list):
            return v
        return ["*"]

    def model_post_init(self, __context):
        """Post-initialization validation and setup"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration consistency"""
        # [MODIFIED] Enforce GEMINI_API_KEYS presence regardless of environment
        # to prevent startup crash in GeminiKeyManager.
        if not self.GEMINI_API_KEYS:
            raise ValueError("At least one Gemini API key is required. Please set GEMINI_API_KEYS in your .env file.")
        
        if self.CACHE_ENABLED and self.CACHE_MAX_SIZE <= 0:
            raise ValueError("CACHE_MAX_SIZE must be positive when caching is enabled")
        
        logger.info(f"Configuration validated for {self.SERVICE_ENVIRONMENT.value} environment")

    def log_configuration(self):
        """Log current configuration (without sensitive data)"""
        logger.info("=== Application Configuration ===")
        logger.info(f"Environment: {self.SERVICE_ENVIRONMENT.value}")
        logger.info(f"Host: {self.SERVICE_HOST}:{self.SERVICE_PORT}")
        logger.info(f"Log Level: {self.SERVICE_LOG_LEVEL.value}")
        logger.info(f"Adapter Security: {'Enabled' if self.SECURITY_ADAPTER_API_KEYS else 'Disabled (INSECURE)'}")
        logger.info(f"Admin Keys: {len(self.SECURITY_ADMIN_API_KEYS)} configured")
        logger.info(f"Gemini Keys: {len(self.GEMINI_API_KEYS)} configured")
        logger.info(f"CORS Origins: {self.SERVICE_CORS_ORIGINS}")
        logger.info(f"Caching: {'Enabled' if self.CACHE_ENABLED else 'Disabled'}")
        if self.CACHE_ENABLED:
            logger.info(f"  - Cache Max Size: {self.CACHE_MAX_SIZE}, TTL: {self.CACHE_TTL}s")
        logger.info("=================================")

_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Get the global configuration instance, loading it if it doesn't exist."""
    global _config
    if _config is None:
        _config = AppConfig()
        _config.log_configuration()
    return _config