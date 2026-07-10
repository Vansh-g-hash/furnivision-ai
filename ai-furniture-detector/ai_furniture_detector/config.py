"""Configuration management for AI Furniture Detector."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Model configuration
    model_path: Path = Field(default_factory=lambda: Path.cwd() / "yolov8x-worldv2.pt")
    default_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    default_iou: float = Field(default=0.70, ge=0.0, le=1.0)
    min_box_area_ratio: float = Field(default=0.00008, ge=0.0, le=1.0)

    # Server configuration
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000)
    web_host: str = Field(default="127.0.0.1")
    web_port: int = Field(default=7860)
    workers: int = Field(default=1, ge=1)

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Paths
    exports_dir: Path = Field(default_factory=lambda: Path.cwd() / "exports")
    debug: bool = Field(default=False)

    class Config:
        env_file = ".env"
        case_sensitive = False

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @validator("exports_dir", pre=True, always=True)
    def ensure_exports_dir(cls, v: Path | str | None) -> Path:
        """Ensure exports directory exists."""
        path = Path(v) if v else Path.cwd() / "exports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary, converting Path objects to strings."""
        data = super().model_dump()
        data["model_path"] = str(self.model_path)
        data["exports_dir"] = str(self.exports_dir)
        return data


# Global settings instance
settings = Settings()
