import cv2
import time
from ultralytics import YOLO

# 1. Load the YOLO model 
model = YOLO('yolov8n.pt')

# 2. Open video source (0 for webcam, or 'video.mp4') 
cap = cv2.VideoCapture(0)

# 3. Setup video writer to save the output 
w, h, fps = (int(cap.get(x)) for x in (3, 4, 5))
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# 4. Track unique object IDs 
unique_ids = set()

# 5. Process the video frame by frame 
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    start = time.time()

    # 6. Run detection with tracking 
    results = model.track(frame, persist=True, imgsz=320, verbose=False)

    # 7. Annotate and count unique IDs 
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().cpu().tolist()
        for obj_id in ids:
            unique_ids.add(obj_id)
        frame = results[0].plot()

    # 8. Display FPS and Count 
    fps_text = f"FPS: {1 / (time.time() - start):.2f}"
    cv2.putText(frame, f"{fps_text} | Objects: {len(unique_ids)}", (20, 50), 0, 1, (0, 255, 0), 2)

    # 9. Save and show 
    out.write(frame)
    cv2.imshow('YOLO Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()