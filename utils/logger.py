# ================================================
# 🪵 utils/logger.py
# ✅ Handles centralized logging for training, inference, and deployment
# ================================================
import logging
import os


class Logger:
    """Centralized logging utility."""

    def __init__(self, name="attractiveness", log_dir="logs"):
        self.name = name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Initialize and configure logger."""
        logger = logging.getLogger(self.name)

        # Prevent duplicate handlers
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # Console Handler
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

        # File Handler
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"{self.name}.log"), mode="a", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def get_logger(self):
        """Return logger instance."""
        return self.logger
