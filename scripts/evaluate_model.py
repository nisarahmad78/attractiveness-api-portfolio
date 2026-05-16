# ================================================================
# 📊 Model Evaluation Script (Class-Based)
# ✅ Evaluates trained model using config/config.json
# ✅ Computes MAE, RMSE, and Accuracy-within tolerance
# Author: Nisar Ahmad
# ================================================================

import os
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from app.model_loader import ModelLoader
from utils.preprocessing import get_transforms
from utils.metrics import mae, rmse, accuracy_within


class ModelEvaluator:
    """Class for evaluating a trained attractiveness model."""

    def __init__(self, config_path: str = "config/config.json"):
        self.cfg = self._load_config(config_path)

        # Paths and Configs
        self.images_dir = self.cfg["data"]["images_dir"]
        self.labels_csv = self.cfg["data"]["labels_csv"]
        self.best_model_path = self.cfg["models"]["best_model_path"]
        self.image_size = self.cfg["training"]["image_size"]
        self.model_name = self.cfg["models"]["model_name"]
        self.seed = self.cfg["training"]["seed"]

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and transforms
        self.model = self._load_model()
        _, self.val_tf = get_transforms(self.image_size)

        # Load dataset
        self.df = pd.read_csv(self.labels_csv)
        self.train_df, self.val_df = train_test_split(self.df, test_size=0.2, random_state=self.seed)

        print(f"✅ Configuration loaded: {config_path}")
        print(f"📦 Model: {self.model_name}")
        print(f"🖼️ Dataset size: {len(self.df)} | Validation: {len(self.val_df)}")
        print(f"💻 Device: {self.device}")

    # ------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------
    def _load_config(self, path: str):
        """Load config.json"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_model(self):
        """Load model with trained weights"""
        loader = ModelLoader(
            model_name=self.model_name,
            weights_path=self.best_model_path,
            device=self.device
        )
        model = loader.load_weights()
        model.to(self.device).eval()
        return model

    # ------------------------------------------------------------
    # Core Evaluation Logic
    # ------------------------------------------------------------
    def evaluate(self, tolerance: float = 0.5, save_results: bool = True):
        """Evaluate model on validation dataset."""
        preds, trues = [], []

        print("\n🚀 Starting evaluation...")
        for _, row in self.val_df.iterrows():
            img_path = os.path.join(self.images_dir, row["filename"])
            if not os.path.exists(img_path):
                print(f"⚠️ Missing: {img_path}")
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                x = self.val_tf(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    pred = self.model(x).clamp(0, 10).item()

                preds.append(pred)
                trues.append(float(row["score"]))
            except Exception as e:
                print(f"❌ Error on {row['filename']}: {e}")

        # Metrics
        mae_score = mae(preds, trues)
        rmse_score = rmse(preds, trues)
        acc_score = accuracy_within(preds, trues, tol=tolerance)

        print(f"\n📊 Evaluation Results:")
        print(f"   → MAE  : {mae_score:.3f}")
        print(f"   → RMSE : {rmse_score:.3f}")
        print(f"   → Within ±{tolerance}: {acc_score:.2%}")

        # Save results
        if save_results:
            result_path = "eval_results.csv"
            pd.DataFrame({
                "filename": self.val_df["filename"],
                "true": trues,
                "pred": preds
            }).to_csv(result_path, index=False)
            print(f"💾 Results saved → {result_path}")

        return {"mae": mae_score, "rmse": rmse_score, "accuracy": acc_score}


# ------------------------------------------------------------
# Main Runner
# ------------------------------------------------------------
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    evaluator = ModelEvaluator(config_path="config/config.json")
    results = evaluator.evaluate(tolerance=0.5)
    print("\n✅ Evaluation complete!")
