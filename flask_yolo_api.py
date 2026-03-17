from flask import Flask, request, jsonify
from ultralytics import YOLO
import io
from PIL import Image

app = Flask(__name__)

# Load the model globally so it stays in RAM
model = YOLO('yolov8n.pt')

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read the uploaded image
    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Run YOLO detection
    results = model(img)
    
    # Extract results into a clean list
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist() # [x1, y1, x2, y2]
            })

    return jsonify({"detections": detections, "total": len(detections)})

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)