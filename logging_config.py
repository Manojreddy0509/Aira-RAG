"""
Logging Configuration Module

Provides structured logging setup for the entire application.
"""
import os
import logging
import sys
from logging.handlers import RotatingFileHandler

# Log directory
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Log level from environment or default to INFO
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


def setup_logging(log_to_file: bool = True):
    """
    Setup application-wide logging
    
    Args:
        log_to_file: Whether to log to file in addition to stdout
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (rotating)
    if log_to_file:
        file_handler = RotatingFileHandler(
            os.path.join(LOG_DIR, "simple_rag.log"),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    root_logger.info(f"Logging initialized at level {LOG_LEVEL}")


# Initialize logging on module import
setup_logging()
