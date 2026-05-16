# ================================================
# 🌟 Attractiveness Prediction Web App (FastAPI)
# ✅ Serves AI model + frontend interface
# ✅ Loads host/port from config.json (no hardcoding)
# ✅ Compatible with AWS, Render, and Local
# ✅ Graceful error handling and static file setup
# Author: Nisar Ahmad
# ================================================

import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.routes.predict import router as predict_router


# ------------------------------------------------
# ⚙️ Config Loader Class
# ------------------------------------------------
class ConfigLoader:
    """Safely loads configuration from a JSON file."""

    def __init__(self, config_path: str = None):
        """Initialize and load config file."""
        # Default config path
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "config.json"
        )

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"❌ Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def get_server_config(self):
        """Return server configuration (host, port, reload)."""
        server_cfg = self.config.get("server", {})
        return {
            "host": server_cfg.get("host", "127.0.0.1"),
            "port": server_cfg.get("port", 8000),
            "reload": server_cfg.get("reload", True),
        }


# ------------------------------------------------
# ⚙️ App Initialization
# ------------------------------------------------
app = FastAPI(
    title="AI Image Attractiveness Predictor",
    description="Web application to predict facial attractiveness scores (0–10) using a trained deep learning model.",
    version="1.0.0",
)

# ------------------------------------------------
# 🌐 CORS Middleware
# ------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------
# 📁 Static Files Setup
# ------------------------------------------------
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)  # prevents startup crash

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ------------------------------------------------
# 🔮 Include Prediction Router
# ------------------------------------------------
app.include_router(predict_router)

# ------------------------------------------------
# 🏠 Home Route (Serves index.html)
# ------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main web interface."""
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h2>⚠️ index.html not found in /static</h2>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

# ------------------------------------------------
# ⚠️ Global Error Handling
# ------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions for clean API responses."""
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "message": "Internal server error. Please check server logs.",
        },
    )

# ------------------------------------------------
# 🚀 Run Command (Reads config or environment)
# ------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    cfg = ConfigLoader()
    server = cfg.get_server_config()

    # 🌍 Allow AWS/Render to override JSON via ENV vars
    host = os.getenv("HOST", server["host"])
    port = int(os.getenv("PORT", server["port"]))
    reload = server["reload"]

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
    )
