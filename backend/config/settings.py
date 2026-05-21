"""
Application Settings Configuration

Uses Pydantic Settings for type-safe configuration management.
All settings are loaded from environment variables or .env file.
"""

from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field
import json


class Settings(BaseSettings):
    """
    Application configuration settings.
    
    All settings can be overridden via environment variables.
    Nested settings use underscore notation (e.g., SMTP_SERVER).
    """
    
    # ===== Database =====
    database_url: str = Field(
        default="sqlite:///./ppe_detection.db",
        description="Database connection URL (SQLite for local dev, PostgreSQL for production)"
    )
    
    # ===== YOLO Model =====
    yolo_model_path: str = Field(
        default="./models/best.pt",
        description="Path to trained YOLO26m weights"
    )
    yolo_confidence_threshold: float = Field(
        default=0.30,
        description="Minimum confidence for YOLO detections"
    )
    yolo_imgsz: int = Field(
        default=640,
        description="Inference image size — must match training resolution (best.pt trained at 640)"
    )
    
    # ===== SAM Model =====
    sam_enabled: bool = Field(
        default=False,
        description="Enable SAM verification (requires GPU). Set to False for YOLO-only mode."
    )
    sam_model_path: str = Field(
        default="./models/sam3.pt",
        description="Path to SAM model weights"
    )
    sam_device: str = Field(
        default="cpu",
        description="Device: cpu or cuda"
    )
    sam_mask_threshold: float = Field(
        default=0.05,
        description="Minimum mask coverage for SAM verification (5%)"
    )
    
    # ===== Person Validation (from diagnostic pipeline) =====
    person_min_coverage: float = Field(
        default=0.10,
        description="Minimum SAM mask coverage to confirm ROI is a person"
    )
    min_person_area: int = Field(
        default=3000,
        description="Minimum pixel area for a person crop (below = too small)"
    )
    aspect_ratio_strict: float = Field(
        default=1.4,
        description="Required h/w ratio for tiny crops (≤50px min dim)"
    )
    aspect_ratio_lenient: float = Field(
        default=0.8,
        description="Required h/w ratio for large crops (≥250px min dim)"
    )
    aspect_dim_low: int = Field(
        default=50,
        description="Min dimension threshold for strict aspect ratio"
    )
    aspect_dim_high: int = Field(
        default=250,
        description="Min dimension threshold for lenient aspect ratio"
    )
    
    # ===== Email (SMTP) =====
    smtp_server: str = Field(
        default="smtp.gmail.com",
        description="SMTP server address"
    )
    smtp_port: int = Field(
        default=587,
        description="SMTP server port"
    )
    sender_email: str = Field(
        default="",
        description="Email address for sending reports"
    )
    sender_password: str = Field(
        default="",
        description="App password for email (not regular password)"
    )
    
    # ===== Reporting =====
    manager_emails: str = Field(
        default='["manager@company.com"]',
        description="JSON array of manager email addresses"
    )
    report_time: str = Field(
        default="23:59",
        description="Time to generate daily reports (HH:MM)"
    )
    report_output_dir: str = Field(
        default="./reports",
        description="Directory for generated PDF reports"
    )
    
    # ===== Violation Tracking =====
    violation_cooldown_seconds: float = Field(
        default=300.0,
        description="Cooldown period (seconds) before re-alerting for same violation. Default 5 minutes."
    )
    violation_iou_threshold: float = Field(
        default=0.3,
        description="Min IoU to match bounding boxes as same person (0.0-1.0)"
    )
    violation_track_timeout: float = Field(
        default=30.0,
        description="Remove person tracks not seen for this duration (seconds)"
    )
    
    # ===== Application =====
    debug: bool = Field(
        default=True,
        description="Enable debug mode"
    )
    cors_origins: str = Field(
        default='["http://localhost:5173", "http://localhost:3000"]',
        description="JSON array of allowed CORS origins"
    )
    
    # ===== Site Configuration =====
    default_site_location: str = Field(
        default="Construction Site A",
        description="Default site location for violations"
    )
    default_camera_id: str = Field(
        default="CAM-001",
        description="Default camera ID for violations"
    )
    
    # ===== ROI Configuration (from thesis) =====
    head_roi_ratio: float = Field(
        default=0.4,
        description="Top portion of person bbox for head ROI"
    )
    torso_roi_start: float = Field(
        default=0.2,
        description="Start of torso ROI (20% from top)"
    )

    # ===== Chatbot / Reporter =====
    openai_api_key: str = Field(
        default="",
        description="OpenAI or OpenRouter API key"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
    
    def get_manager_emails_list(self) -> List[str]:
        """Parse manager emails from JSON string to list."""
        try:
            return json.loads(self.manager_emails)
        except json.JSONDecodeError:
            return [self.manager_emails]  # Single email as fallback
    
    def get_cors_origins_list(self) -> List[str]:
        """Parse CORS origins from JSON string to list."""
        try:
            return json.loads(self.cors_origins)
        except json.JSONDecodeError:
            return ["http://localhost:5173"]


# Global settings instance
settings = Settings()
