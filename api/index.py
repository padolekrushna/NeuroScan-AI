"""
FastAPI backend for Brain MRI Tumor Classification.
Serves the TensorFlow CNN model via a REST API.
"""

import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(
    title="Brain Tumor Detection API",
    description="AI-powered brain MRI tumor classification using CNN",
    version="2.0.0",
)

# CORS — allow React dev server and production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ──────────────────────────────────────────────
IMG_SIZE = 224
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ── Model Loading ──────────────────────────────────────────
model = None

def get_model():
    """Lazy-load the TensorFlow model."""
    global model
    if model is None:
        import tensorflow as tf
        model_path = os.path.join(os.path.dirname(__file__), "..", "model", "model.h5")
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model file not found at {model_path}. "
                "Please train the model first using the notebook in model/model_building.ipynb"
            )
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
    return model


# ── Routes ─────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    """
    Accept an MRI image upload, run inference, and return prediction results.

    Returns JSON:
    {
        "prediction": "Glioma",
        "confidence": 97.32,
        "probabilities": {
            "Glioma": 97.32,
            "Meningioma": 1.45,
            "No Tumor": 0.12,
            "Pituitary": 1.11
        }
    }
    """
    # Validate content type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    # Read and validate size
    contents = await image.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    try:
        # Preprocess
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        loaded_model = get_model()
        preds = loaded_model.predict(img_array)[0]
        class_index = int(np.argmax(preds))
        confidence = float(preds[class_index] * 100)

        probabilities = {
            name: round(float(preds[i] * 100), 2)
            for i, name in enumerate(CLASS_NAMES)
        }

        return {
            "prediction": CLASS_NAMES[class_index],
            "confidence": round(confidence, 2),
            "probabilities": probabilities,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ── Local dev entry point ──────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)
