
#  backend.py
#  Flask API for Corn Pest Detection


from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import cv2

app = Flask(__name__)
CORS(app)  # Required to allow React on port 5173 to call this server


#  MODEL CONFIGURATION



MODEL_PATH = "final_corn_pest_model.h5"

# Class names 
# Index 0 = Corn Borers, 1 = Corn Earworms, 2 = Healthy Corn, 3 = rootworm
# These IDs match the keys in src/data/mockData.js in the React app
CLASS_NAMES = ["borer", "earworm", "healthy", "rootworm"]

#  model input is (224, 224, 3)
IMAGE_SIZE = (224, 224)

# ==============================================================
#  LOAD MODEL AT STARTUP
# ==============================================================
print("=" * 50)
print(f"Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")
print(f"   Input shape  : {model.input_shape}")
print(f"   Output shape : {model.output_shape}")
print(f"   Classes      : {CLASS_NAMES}")
print("=" * 50)



#  IMAGE PREPROCESSING

def preprocess_image(file_bytes):
    # Open image and force RGB (handles PNG transparency, grayscale, etc.)
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # Resize 
    img = img.resize(IMAGE_SIZE)

    # Convert to float32 numpy array, shape: (H, W, 3)
    img_array = np.array(img, dtype=np.float32)

   
   
   
    # NORMALIZATION 
   
    img_array = img_array
    
    

    # Add batch dimension → shape: (1, H, W, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ==============================================================
#  GRAD-CAM HEATMAP GENERATION
#  Produces explainability visualizations shown in the frontend.
# ==============================================================
def get_last_conv_layer_name():
    """Automatically finds the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def generate_gradcam(file_bytes, predicted_index):
    """
    Generates Grad-CAM heatmap and overlay for the predicted class.
    Returns (heatmap_base64, overlay_base64) or (None, None) on failure.
    """
    try:
        # Prepare image tensor
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_resized = img.resize(IMAGE_SIZE)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)

        # Find last conv layer automatically
        last_conv_name = get_last_conv_layer_name()
        if last_conv_name is None:
            print("⚠️  No Conv2D layer found — skipping Grad-CAM")
            return None, None

        # Build gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[
                model.get_layer(last_conv_name).output,
                model.output
            ]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, predicted_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()

        # Weight feature maps by gradients
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        # Produce heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        heatmap = np.uint8(255 * heatmap)

        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
        heatmap_color   = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        # Convert original image to BGR for OpenCV overlay
        original_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

        # Overlay: blend heatmap onto original image
        overlay = cv2.addWeighted(original_bgr, 0.55, heatmap_color, 0.45, 0)

        # Encode heatmap as base64 PNG
        _, heatmap_buf = cv2.imencode(".png", heatmap_color)
        heatmap_b64 = base64.b64encode(heatmap_buf).decode("utf-8")

        # Encode overlay as base64 PNG
        _, overlay_buf = cv2.imencode(".png", overlay)
        overlay_b64 = base64.b64encode(overlay_buf).decode("utf-8")

        return heatmap_b64, overlay_b64

    except Exception as e:
        print(f"⚠️  Grad-CAM failed: {e}")
        return None, None



#  predict  — MAIN ENDPOINT
#  Called by the React frontend (src/services/api.js)

@app.route("/predict", methods=["POST"])
def predict():
    # ── Validate incoming request ──────────────────────────────
    if "file" not in request.files:
        return jsonify({
            "error": "No file received. Send image as multipart/form-data with field name 'file'."
        }), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "File field is empty."}), 400

    try:
        file_bytes = file.read()

        # ── Preprocess & predict ───────────────────────────────
        img_tensor = preprocess_image(file_bytes)
        raw_output = model.predict(img_tensor, verbose=0)  # shape: (1, 4)

        #  model's last layer is Dense(4, activation='softmax')
        # so it already outputs probabilities — use directly, no extra softmax
        probabilities = raw_output[0]

        predicted_index = int(np.argmax(probabilities))
        confidence      = float(probabilities[predicted_index]) * 100
        predicted_class = CLASS_NAMES[predicted_index]

        # Build probabilities dict — values are 0–100 percentages
        probs_dict = {
            CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        # ── Generate Grad-CAM visualizations ──────────────────
        heatmap_b64, overlay_b64 = generate_gradcam(file_bytes, predicted_index)

        # ── Build and return final response ───────────────────
        response = {
            "predicted_class": predicted_class,
            "confidence":      round(confidence, 2),
            "probabilities":   probs_dict,
            "gradcam_heatmap": heatmap_b64,   # base64 PNG or null
            "gradcam_overlay": overlay_b64,   # base64 PNG or null
        }

        print(f"✅ Predicted: {predicted_class}  |  Confidence: {confidence:.1f}%")
        return jsonify(response), 200

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


# ==============================================================
#  /health  — STATUS CHECK
#  Open http://localhost:5000/health in browser to verify server
# ==============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":     "online",
        "model":      MODEL_PATH,
        "classes":    CLASS_NAMES,
        "image_size": list(IMAGE_SIZE),
    }), 200


# ==============================================================
#  START SERVER
# ==============================================================
if __name__ == "__main__":
    print("\n🌽 CornGuard AI backend running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
