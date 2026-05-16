# ================================================================
# 🧩 Logger Utility (Class-Based)
# Provides a consistent, configurable logging setup across the app.
# Author: Nisar Ahmad
# ================================================================

import logging
import os


class AppLogger:
    """
    Centralized logging utility class.

    Example:
        from app.utils.logger import AppLogger
        logger = AppLogger("trainer").get_logger()
        logger.info("Training started!")
    """

    def __init__(self, name: str = "app", log_dir: str = "logs"):
        self.name = name
        self.log_dir = os.path.join(os.getcwd(), log_dir)
        self.log_file = os.path.join(self.log_dir, f"{self.name}.log")
        os.makedirs(self.log_dir, exist_ok=True)
        self._logger = self._setup_logger()

    def _setup_logger(self):
        """Configure and return a logger instance."""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)

        # Prevent duplicate handlers on reloads
        if not logger.handlers:
            # File Handler
            file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            file_handler.setLevel(logging.INFO)

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Formatter
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger

    def get_logger(self):
        """Return the configured logger."""
        return self._logger


# ------------------------------------------------------------
# ✅ Shortcut Function (backward-compatible)
# ------------------------------------------------------------
def get_logger(name: str = "app"):
    """Return a preconfigured logger (for backward compatibility)."""
    return AppLogger(name).get_logger()
