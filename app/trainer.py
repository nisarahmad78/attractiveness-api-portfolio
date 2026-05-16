# ================================================================
# 📘 Attractiveness Model Trainer (Class-Based)
# ✅ Uses config/config.json
# ✅ Supports resume, checkpoints, AMP, CPU/GPU
# ✅ Clean OOP version for production
# Author: Nisar Ahmad
# ================================================================

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import contextlib
import numpy as np
import pandas as pd
import imghdr
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import models, transforms
from app.utils.logger import get_logger
from app.datasets.face_dataset import FaceDataset


class AttractivenessTrainer:
    """Handles training, validation, and checkpointing."""

    def __init__(self, config_path: str = "config/config.json"):
        self.logger = get_logger("trainer")

        # ------------------------------------------------------
        # Load Config
        # ------------------------------------------------------
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"⚠️ Config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available()

        self._load_config_values()
        self._set_seeds()
        self._ensure_dirs()

        self.logger.info(f"📦 Loaded config from {config_path}")
        self.logger.info(f"🔧 Using device: {self.device}")
        self.logger.info(f"🧩 Model: {self.model_name}")

    # ----------------------------------------------------------
    # 🔧 Setup helpers
    # ----------------------------------------------------------
    def _load_config_values(self):
        c = self.cfg
        self.data_dir = c["data"]["images_dir"]
        self.csv_path = c["data"]["labels_csv"]
        self.ckpt_path = c["models"]["checkpoint_dir"]
        self.best_model_path = c["models"]["best_model_path"]
        self.model_name = c["models"]["model_name"]

        self.batch_size = c["training"]["batch_size"]
        self.epochs = c["training"]["epochs"]
        self.lr = c["training"]["lr"]
        self.img_size = c["training"]["image_size"]
        self.num_workers = c["training"]["num_workers"]
        self.seed = c["training"]["seed"]

    def _set_seeds(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _ensure_dirs(self):
        for path in [self.ckpt_path, self.best_model_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    # ----------------------------------------------------------
    # 📂 Dataset Loader
    # ----------------------------------------------------------
    def _validate_dataset(self):
        df = pd.read_csv(self.csv_path)
        if "filename" not in df.columns or "score" not in df.columns:
            raise ValueError("CSV must contain 'filename' and 'score' columns.")

        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).clip(0, 10)

        missing, bad = [], []
        for name in tqdm(df["filename"], desc="🔍 Checking images"):
            path = os.path.join(self.data_dir, name)
            if not os.path.exists(path):
                missing.append(name)
            else:
                try:
                    if imghdr.what(path) is None:
                        bad.append(name)
                except:
                    bad.append(name)

        if missing or bad:
            self.logger.warning(f"⚠️ Missing: {len(missing)}, Corrupted: {len(bad)}")
            df = df[~df["filename"].isin(set(missing + bad))].reset_index(drop=True)
            valid_csv = os.path.join(os.path.dirname(self.csv_path), "labels_validated.csv")
            df.to_csv(valid_csv, index=False)
            self.logger.info(f"✅ Saved validated CSV → {valid_csv}")
        else:
            self.logger.info("✅ All images OK")

        return train_test_split(df, test_size=0.2, random_state=self.seed)

    # ----------------------------------------------------------
    # 🧩 Model Builder
    # ----------------------------------------------------------
    def _build_model(self):
        if self.model_name == "mobilenet_v2":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        elif self.model_name == "efficientnet_b3":
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        elif self.model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Linear(model.fc.in_features, 1)
        else:
            raise ValueError("Unsupported model name in config.json")

        return model.to(self.device)

    # ----------------------------------------------------------
    # 📊 Evaluation
    # ----------------------------------------------------------
    def _evaluate(self, model, loader, autocast):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.view(-1, 1).to(self.device)
                with autocast():
                    o = model(x)
                preds += o.squeeze(1).cpu().tolist()
                trues += y.squeeze(1).cpu().tolist()
        preds, trues = np.array(preds), np.array(trues)
        mae = np.mean(np.abs(preds - trues))
        rmse = np.sqrt(np.mean((preds - trues) ** 2))
        return mae, rmse

    # ----------------------------------------------------------
    # 🚀 Train
    # ----------------------------------------------------------
    def train(self):
        train_df, val_df = self._validate_dataset()

        train_tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        train_loader = DataLoader(
            FaceDataset(train_df, self.data_dir, train_tf, self.img_size),
            batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        val_loader = DataLoader(
            FaceDataset(val_df, self.data_dir, val_tf, self.img_size),
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        model = self._build_model()
        criterion = nn.HuberLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        if self.use_amp:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
        else:
            autocast = contextlib.nullcontext
            scaler = None

        best_mae = float("inf")
        start_epoch = 0

        # Resume training
        if os.path.exists(self.ckpt_path):
            try:
                ckpt = torch.load(self.ckpt_path, map_location=self.device)
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                scheduler.load_state_dict(ckpt["scheduler_state"])
                best_mae = ckpt["best_mae"]
                start_epoch = ckpt["epoch"] + 1
                self.logger.info(f"🔁 Resumed from epoch {start_epoch}, best MAE={best_mae:.3f}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to resume: {e}")

        # Training loop
        for epoch in range(start_epoch, self.epochs):
            model.train()
            total_loss = 0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for xb, yb in loop:
                xb, yb = xb.to(self.device), yb.view(-1, 1).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    out = model(xb)
                    loss = criterion(out, yb)

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * xb.size(0)
                loop.set_postfix(loss=total_loss / len(train_loader.dataset))

            mae, rmse = self._evaluate(model, val_loader, autocast)
            scheduler.step(mae)
            self.logger.info(f"📈 Epoch {epoch+1}: MAE={mae:.3f}, RMSE={rmse:.3f}")

            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_mae": best_mae
            }, self.ckpt_path)

            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), self.best_model_path)
                self.logger.info(f"🏆 Best model updated (MAE={best_mae:.3f})")

        self.logger.info(f"✅ Training complete! Best MAE: {best_mae:.3f}")


# ----------------------------------------------------------
# 🏁 Entry Point
# ----------------------------------------------------------
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    trainer = AttractivenessTrainer()
    trainer.train()
