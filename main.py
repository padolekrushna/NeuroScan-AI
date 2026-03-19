import matplotlib
matplotlib.use("Agg")

from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt
import uuid
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "model/model.h5"
IMG_SIZE = 224
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("DONE: Model loaded successfully")
except Exception as e:
    print(f"INFO: Model not found or error loading ({e}). App will run in 'training' mode.")
    model = None


@app.route("/")
def index():
    return render_template("index.html", is_training=(model is None))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return render_template("index.html", error="The model is currently training. Please check back in a few minutes.", is_training=True)

        # Validate file upload
        if "image" not in request.files:
            return render_template("index.html", error="No file uploaded")
        
        file = request.files["image"]
        
        if file.filename == "":
            return render_template("index.html", error="No file selected")
        
        # Check file extension
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return render_template("index.html", error="Invalid file type. Please upload an image.")

        # Generate unique filename
        filename = f"{uuid.uuid4()}{file_ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Process image
        image = Image.open(filepath).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        preds = model.predict(img_array)[0]
        class_index = np.argmax(preds)
        confidence = preds[class_index] * 100

        # Create enhanced chart with modern styling
        chart_filename = f"chart_{uuid.uuid4()}.png"
        chart_path = os.path.join(app.config["UPLOAD_FOLDER"], chart_filename)
        
        plt.figure(figsize=(10, 6))
        colors = ['#667eea' if i == class_index else '#e2e8f0' for i in range(len(CLASS_NAMES))]
        bars = plt.bar(CLASS_NAMES, preds * 100, color=colors, edgecolor='#2d3748', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.ylabel("Confidence (%)", fontsize=12, fontweight='bold')
        plt.title("Class-wise Prediction Confidence", fontsize=14, fontweight='bold', pad=20)
        plt.ylim(0, 105)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return render_template(
            "result.html",
            prediction=CLASS_NAMES[class_index],
            confidence=f"{confidence:.2f}",
            image_path=filepath,
            chart_path=chart_path
        )

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Clean up uploaded file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return render_template("index.html", error=f"An error occurred: {str(e)}")


# Optional: Add a cleanup route to remove old files
@app.route("/cleanup")
def cleanup():
    try:
        files_removed = 0
        current_time = datetime.now().timestamp()
        
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file_age = current_time - os.path.getmtime(filepath)
            
            # Remove files older than 1 hour (3600 seconds)
            if file_age > 3600:
                os.remove(filepath)
                files_removed += 1
        
        return jsonify({
            "status": "success",
            "files_removed": files_removed
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })


# Error handlers
@app.errorhandler(413)
def too_large(e):
    return render_template("index.html", error="File is too large. Maximum size is 10MB.")


@app.errorhandler(404)
def not_found(e):
    return render_template("index.html")


@app.errorhandler(500)
def internal_error(e):
    return render_template("index.html", error="An internal error occurred. Please try again.")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)