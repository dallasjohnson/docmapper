from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for the application"""
    MODEL_NAME: str = "gemini-2.0-flash-exp"
    TEXT_EMBEDDING_MODEL_ID: str = "text-embedding-004"
    DPI: int = 300
    