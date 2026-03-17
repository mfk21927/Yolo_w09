import os
import json
from ultralytics import YOLO
import cv2

IMAGES_DIR = "sample_images"
OUTPUT_DIR = "detected_images"
RESULTS_FILE = "detection_results.json"

# supported image extensions
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_images_from_folder():
    # get all image files from the sample_images directory
    if not os.path.exists(IMAGES_DIR):
        print(f"Folder '{IMAGES_DIR}' not found. Run download_images.py first.")
        return []

    image_paths = [
        os.path.join(IMAGES_DIR, f)
        for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    if not image_paths:
        print(f"No images found in '{IMAGES_DIR}'. Run download_images.py first.")

    return image_paths


def run_detection(model, image_paths):
    results_data = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for image_path in image_paths:
        print(f"Running detection on: {image_path}")
        results = model(image_path)

        image = cv2.imread(image_path)
        image_results = {
            "image": image_path,
            "detections": []
        }

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                image_results["detections"].append({
                    "class": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    }
                })

                # draw bounding box on image
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # draw label with confidence
                label = f"{class_name} {confidence:.2f}"
                label_y = int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + 20
                cv2.putText(image, label, (int(x1), label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # save annotated image
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
        cv2.imwrite(output_path, image)
        print(f"Saved annotated image: {output_path}")

        results_data.append(image_results)

    return results_data


def generate_summary(results_data):
    total_objects = 0
    class_counts = {}

    for image_result in results_data:
        for det in image_result["detections"]:
            total_objects += 1
            cls = det["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

    summary = {
        "total_objects_detected": total_objects,
        "classes_detected": class_counts,
        "images_processed": len(results_data)
    }
    return summary


def save_results(results_data, summary):
    output = {
        "summary": summary,
        "results": results_data
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Detection results saved to: {RESULTS_FILE}")


def main():
    # load pre-trained yolov8 model
    model = YOLO("yolov8n.pt")

    # load images from sample_images folder
    image_paths = load_images_from_folder()
    if not image_paths:
        return

    # run detection on all images
    results_data = run_detection(model, image_paths)

    # generate summary
    summary = generate_summary(results_data)

    # print summary to console
    print("\n===== Detection Summary =====")
    print(f"Images processed   : {summary['images_processed']}")
    print(f"Total objects found: {summary['total_objects_detected']}")
    print("Classes detected:")
    for cls, count in summary["classes_detected"].items():
        print(f"  {cls}: {count}")

    # save results to json
    save_results(results_data, summary)


if __name__ == "__main__":
    main()
