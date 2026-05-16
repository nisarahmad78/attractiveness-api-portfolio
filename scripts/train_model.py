# ================================================================
# 🚀 Training Launcher (Class-Based)
# ✅ Safe, modular version of training entry point
# ✅ Uses app.trainer.AttractivenessTrainer
# ✅ Run with:  python -m scripts.train_model
# Author: Nisar Ahmad
# ================================================================

import os
import sys
import importlib


class TrainingLauncher:
    """Class-based launcher for the attractiveness model trainer."""

    def __init__(self):
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        print(f"🧩 Project root added to path: {self.root}")

    def run(self):
        """Import and start the training process."""
        print("🚀 Starting Training ...")

        try:
            # Dynamically import the trainer module
            trainer_module = importlib.import_module("app.trainer")

            # Support both class-based and function-based trainers
            if hasattr(trainer_module, "AttractivenessTrainer"):
                trainer_class = getattr(trainer_module, "AttractivenessTrainer")
                trainer = trainer_class()
                trainer.train_from_config()
            elif hasattr(trainer_module, "train_from_config"):
                trainer_module.train_from_config()
            else:
                raise AttributeError("Trainer module does not have a valid entry point.")

            print("✅ Training completed successfully.")

        except Exception as e:
            print("❌ Error during training:", e)


# ------------------------------------------------------------
# Main Entry
# ------------------------------------------------------------
if __name__ == "__main__":
    launcher = TrainingLauncher()
    launcher.run()
