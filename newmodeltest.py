# tf_live_ripe.py
# ==============================
# Live Ripe Tomato Detection using TFLite YOLOv8
# ==============================

from ultralytics import YOLO

# ==============================
# Load TFLite Model
# ==============================
model = YOLO("best_float16.tflite")  # Path to your TFLite model
print("Model Loaded ✅")

# ==============================
# Start Live Camera Detection
# ==============================
print("Starting Camera... Press Q to Quit")
# classes=[2] ensures we only detect 'ripe' tomatoes (class 2 in your dataset)
model.predict(
    source=0,       # 0 = default webcam
    classes=[2],    # Only ripe tomatoes
    conf=0.25,      # Minimum confidence
    show=True       # Display live annotated video
)
print("Camera Closed ✅")
