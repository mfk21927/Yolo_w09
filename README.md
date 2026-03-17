
# ML_WEEK09_B01
# 🚀 ML-Internship

> **Name:** Muhammad Fahad 
> **Email:** [![Email](https://img.shields.io/badge/Email-mfk21927@gmail.com-red?style=flat-square&logo=gmail&logoColor=white)](mailto:mfk21927@gmail.com) 
> **LinkedIn:** [![LinkedIn](https://img.shields.io/badge/LinkedIn-Muhammad%20Fahad-blue?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammad-fahad-087057293) 
> **Start Date:** 20-12-2025 

---

![Internship](https://img.shields.io/badge/Status-Active-blue?style=for-the-badge)
![Batch](https://img.shields.io/badge/Batch-B01-orange?style=for-the-badge)
[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-ORM-red?logo=python&logoColor=white)](https://www.sqlalchemy.org/)
[![TFLite](https://img.shields.io/badge/TFLite-ai--edge--litert-orange?logo=tensorflow&logoColor=white)](https://ai.google.dev/edge/litert)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview

This repository documents my **Week 9 Machine Learning Internship tasks**, focused on **Advanced Object Detection with YOLOv8**.
It covers the entire pipeline from **environment setup and custom model training** to **real-time video tracking and REST API deployment** for computer vision applications.

-----

## 📈 Week 9 Tasks Overview

| Task | Title | Key Tech | Status |
| :--- | :--- | :--- | :--- |
| 9.1 | YOLO Setup & Basic Detection | Ultralytics, JSON, OpenCV | ✅ Completed |
| 9.2 | Custom YOLO Training | Roboflow, mAP, Export (.onnx) | ✅ Completed |
| 9.3 | Real-time Video Detection | Object Tracking, FPS, cv2 | ✅ Completed |
| 9.4 | YOLO Model Deployment in Flask | REST API, JSON, Postman | ✅ Completed |

-----

## ✅ Task Details

### **Task 9.1: YOLO Setup & Basic Object Detection**

  - **File:** `yolo_detection.py`
  - **Implementation:**
      - Configured the environment with `ultralytics` and loaded the pre-trained `yolov8n.pt` model.
      - Performed object detection on sample images, extracting bounding boxes and confidence scores.
      - Automated the generation of a **summary report** and a **JSON file** containing all detection results.
      - Visualized results by drawing annotated boxes using OpenCV.

### **Task 9.2: Custom YOLO Training**

  - **File:** `yolo_custom_training.py`
  - **Implementation:**
      - Prepared a custom dataset using **Roboflow** in YOLO format (images and labels).
      - Configured `data.yaml` and trained the model for **50 epochs**.
      - Evaluated performance using **mAP (mean Average Precision)** and analyzed training curves.
      - Exported the final trained model to **ONNX** and **TFLite** formats for cross-platform compatibility.

### **Task 9.3: Real-time Video Object Detection**

  - **File:** `yolo_video_detection.py`
  - **Implementation:**
      - Developed a real-time processing script for webcam and video files.
      - Implemented **object tracking** to assign unique IDs to detected objects across frames.
      - Created a unique object counter and optimized performance for higher **FPS** on 8GB RAM hardware.
      - Saved the output video with live annotations and ID tracking.

### **Task 9.4: YOLO Model Deployment in Flask**

  - **File:** `flask_yolo_api.py`
  - **Implementation:**
      - Deployed the YOLOv8 model as a **REST API** using Flask.
      - Created endpoints for both **image upload** and **real-time video streaming**.
      - Optimized the API with global model loading and a threading queue for handling multiple requests.
      - Verified JSON responses (classes, boxes, and confidences) using **Postman**.

-----

## 📁 Project Structure

```
ML_WEEK09_B01/
├── yolo_detection.py         # Task 9.1 — Basic detection & JSON export
├── yolo_custom_training.py   # Task 9.2 — Custom dataset training script
├── yolo_video_detection.py   # Task 9.3 — Real-time tracking & FPS optimization
├── flask_yolo_api.py         # Task 9.4 — YOLO REST API deployment
├── data.yaml                 # Dataset configuration file
├── best.pt                   # Custom trained weights
└── output_detection.mp4      # Sample processed video
```

-----

## 💻 Tech Stack

  * **Languages:** Python 3.10
  * **Frameworks:** Flask, Ultralytics (YOLOv8)
  * **Libraries:** OpenCV, PyTorch, NumPy, Pillow
  * **Tools:** Roboflow, Postman, Git, VS Code

-----
