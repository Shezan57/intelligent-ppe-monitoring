"""
Database Connection Module

Provides SQLAlchemy engine, session factory, and dependency injection.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from config.settings import settings


# Create database engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # Enable connection health checks
    pool_size=5,
    max_overflow=10,
    echo=settings.debug  # Log SQL queries in debug mode
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency for FastAPI.
    
    Usage:
        @app.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    
    Yields:
        SQLAlchemy Session instance
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
