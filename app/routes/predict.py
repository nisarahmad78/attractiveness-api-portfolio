import os
import uuid
import time
import json
import torch
import numpy as np
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import onnxruntime as ort

from app.model_loader import ModelLoader
from app.utils.preprocessing import get_transforms
from app.utils.logger import get_logger

# ------------------------------------------------
# ⚙️ Setup
# ------------------------------------------------
logger = get_logger()
router = APIRouter()


# ------------------------------------------------
# 🧩 Predictor Class
# ------------------------------------------------
class Predictor:
    """Handles model loading, YuNet face detection, gender classification, and prediction logic."""

    def __init__(self, config_path: str = "config/config.json"):
        # ------------------------------------------------
        # 🧾 Load Config
        # ------------------------------------------------
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"⚠️ Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.model_cfg = self.cfg["models"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.transforms = None

        # Temp dir for uploads
        self.temp_dir = "temp_upload"
        os.makedirs(self.temp_dir, exist_ok=True)

        # ------------------------------------------------
        # 🧠 Base Directory
        # ------------------------------------------------
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # ------------------------------------------------
        # 🧩 Face Detection (YuNet)
        # ------------------------------------------------
        face_cfg = self.model_cfg["face_detector"]
        self.yunet_path = os.path.join(base_dir, face_cfg["model_path"])

        if not os.path.exists(self.yunet_path):
            raise FileNotFoundError(f"❌ YuNet model not found: {self.yunet_path}")

        self.face_detector = cv2.FaceDetectorYN_create(
            model=self.yunet_path,
            config='',
            input_size=(320, 320),
            score_threshold=face_cfg.get("score_threshold", 0.6),
            nms_threshold=face_cfg.get("nms_threshold", 0.3),
            top_k=face_cfg.get("top_k", 5000),
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )

        # ------------------------------------------------
        # 🚻 Gender Classification Model (ONNXRuntime)
        # ------------------------------------------------
        self.gender_session = None
        gender_model_path = os.path.join(base_dir, self.model_cfg.get("gender_model_path", ""))

        if os.path.exists(gender_model_path):
            try:
                self.gender_session = ort.InferenceSession(
                    gender_model_path, providers=["CPUExecutionProvider"]
                )
                logger.info(f"✅ Gender classifier loaded successfully (ONNXRuntime): {gender_model_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load gender model: {e}")
        else:
            logger.warning("⚠️ Gender model not found — skipping gender detection.")

    # ------------------------------------------------
    # 🔧 Initialize Attractiveness Model
    # ------------------------------------------------
    def initialize_model(self):
        """Initialize MobileNet V2 attractiveness model and preprocessing."""
        if self.model is None:
            try:
                loader = ModelLoader(
                    model_name=self.model_cfg.get("model_name", "mobilenet_v2"),
                    weights_path=self.model_cfg["best_model_path"],
                    device=self.device
                )
                self.model = loader.load_weights()
                _, self.transforms = get_transforms(self.cfg["training"]["image_size"])
                logger.info(
                    f"✅ Model '{self.model_cfg.get('model_name')}' loaded successfully on {self.device.upper()}"
                )
            except Exception as e:
                logger.error(f"❌ Model initialization failed: {e}")
                raise HTTPException(status_code=500, detail="Model failed to load.")

    # ------------------------------------------------
    # 🚻 Gender Classification (ONNXRuntime)
    # ------------------------------------------------
    def classify_gender(self, face_crop: np.ndarray):
        """Classify gender using ONNXRuntime model (binary output sigmoid)."""
        if self.gender_session is None:
            return None, None

        try:
            img = cv2.resize(face_crop, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # CHW
            img = np.expand_dims(img, 0)        # BCHW
            inputs = {self.gender_session.get_inputs()[0].name: img}

            preds = self.gender_session.run(None, inputs)[0]
            prob_female = float(preds[0][0])
            gender_label = "Female" if prob_female > 0.5 else "Male"
            confidence = prob_female if gender_label == "Female" else 1 - prob_female
            return gender_label, confidence
        except Exception as e:
            logger.warning(f"⚠️ Gender classification failed: {e}")
            return None, None

    # ------------------------------------------------
    # 🧮 Single Image Prediction (Multi-face Supported)
    # ------------------------------------------------
    def predict_single(self, file: UploadFile) -> dict:
        """Predict attractiveness for all detected faces in an image."""
        filename = file.filename or f"{uuid.uuid4().hex}.jpg"
        tmp_path = os.path.join(self.temp_dir, filename)

        try:
            # Save temporarily
            with open(tmp_path, "wb") as outp:
                outp.write(file.file.read())

            # Read image
            image_cv = cv2.imread(tmp_path)
            if image_cv is None:
                return {"filename": filename, "error": "invalid image format"}

            # Resize large images
            max_side = max(image_cv.shape[:2])
            if max_side > 640:
                scale = 640.0 / max_side
                image_cv = cv2.resize(
                    image_cv,
                    (int(image_cv.shape[1] * scale), int(image_cv.shape[0] * scale))
                )

            h, w, _ = image_cv.shape
            self.face_detector.setInputSize((w, h))

            # Detect all faces
            faces = self.face_detector.detect(image_cv)
            if isinstance(faces, tuple):
                faces = faces[1]

            if faces is None or len(faces) == 0:
                logger.warning(f"⚠️ No face detected in {filename}")
                return {
                    "filename": filename,
                    "face_detected": False,
                    "message": "No faces detected — skipped"
                }

            # List to hold all face results
            face_results = []

            # ✅ Iterate through each detected face
            for idx, face in enumerate(faces):
                x, y, w, h = map(int, face[:4])
                face_crop = image_cv[y:y + h, x:x + w]

                # Gender classification
                gender, confidence = self.classify_gender(face_crop)

                # Preprocess for attractiveness model
                img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                x_tensor = self.transforms(img).unsqueeze(0).to(self.device)

                # Predict attractiveness
                with torch.no_grad():
                    y_pred = self.model(x_tensor).clamp(0, 10).item()

                # ✅ Apply linear calibration for all faces
                fitted = 2.6639 * y_pred - 14.9283
                fitted_constrained = float(np.clip(fitted, 0.0, 10.0))

                # Add to results
                face_results.append({
                    "face_index": idx,
                    "bbox": [x, y, w, h],
                    "observed": float(y_pred),
                    "fitted": float(fitted),
                    "fitted_constrained": fitted_constrained,
                    "gender": gender,
                    "gender_confidence": confidence
                })

            # Return all faces
            return {
                "filename": filename,
                "num_faces": len(face_results),
                "faces": face_results,
                "face_detected": True
            }

        except Exception as e:
            logger.error(f"❌ Prediction failed for {filename}: {e}")
            return {"filename": filename, "error": str(e)}

        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # ------------------------------------------------
    # 📦 Batch Prediction
    # ------------------------------------------------
    async def predict_batch(self, files: list[UploadFile]):
        """Predict multiple images at once."""
        self.initialize_model()

        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        start_time = time.perf_counter()
        results = []

        for f in files:
            result = self.predict_single(f)
            results.append(result)

        elapsed = time.perf_counter() - start_time
        logger.info(f"Batch completed: {len(files)} files in {elapsed:.2f}s")

        return {"results": results, "batch_elapsed_seconds": elapsed}


# ------------------------------------------------
# 🚀 FastAPI Route
# ------------------------------------------------
predictor = Predictor()


@router.post("/predict")
async def predict_route(files: list[UploadFile] = File(...)):
    """API route for batch prediction."""
    return await predictor.predict_batch(files)
