"""Poultry detection and disease classification pipeline.

Two YOLOv8 detectors find whole birds and neck regions; two ResNet-50
classifiers label each crop. Models are loaded lazily on first use, so the
web pages work (and the tests run) without the weights or PyTorch installed.

Weights are looked up in CHARM_MODEL_DIR (default: <repo>/webapp/models):
    yolo_body.pt     YOLOv8 whole-bird detector
    yolo_neck.pt     YOLOv8 neck-region detector
    resnet50_body.pth  ResNet-50, 3 classes: Marek, Disease, Healthy
    resnet50_neck.pth  ResNet-50, 3 classes: infectious_coryza, fowl_pox, Healthy
Each path can also be overridden individually (see MODEL_PATHS below).
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

MODEL_DIR = Path(os.getenv("CHARM_MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
MODEL_PATHS = {
    "yolo_body": Path(os.getenv("CHARM_YOLO_BODY", MODEL_DIR / "yolo_body.pt")),
    "yolo_neck": Path(os.getenv("CHARM_YOLO_NECK", MODEL_DIR / "yolo_neck.pt")),
    "resnet_body": Path(os.getenv("CHARM_RESNET_BODY", MODEL_DIR / "resnet50_body.pth")),
    "resnet_neck": Path(os.getenv("CHARM_RESNET_NECK", MODEL_DIR / "resnet50_neck.pth")),
}

# Class labels full body
CLASS_LABELS = {
    0: "Marek",
    1: "Disease",
    2: "Healthy"
}
# Class labels neck area
CLASS_LABELS1 = {
    0: "infectious_coryza",
    1: "fowl_pox",
    2: "Healthy"
}

_models = None
_models_lock = threading.Lock()


def _load_models():
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as transforms
    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def resnet(path):
        net = tv_models.resnet50(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, 3)  # 3 classes
        net.load_state_dict(torch.load(path, map_location=device))
        return net.to(device).eval()

    return SimpleNamespace(
        torch=torch,
        device=device,
        body=resnet(MODEL_PATHS["resnet_body"]),
        neck=resnet(MODEL_PATHS["resnet_neck"]),
        yolo=YOLO(str(MODEL_PATHS["yolo_body"])),
        yolo_neck=YOLO(str(MODEL_PATHS["yolo_neck"])),
        # Must match training preprocessing
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    )


def get_models():
    """Load the four models once, on first use."""
    global _models
    if _models is None:
        with _models_lock:
            if _models is None:
                missing = [str(p) for p in MODEL_PATHS.values() if not p.exists()]
                if missing:
                    raise FileNotFoundError(
                        "Model weights not found: " + ", ".join(missing)
                        + ". Place them in CHARM_MODEL_DIR (see README).")
                _models = _load_models()
    return _models


QUALITY_THRESHOLDS = {
    "blur": 50.0,             # Adjust based on how sharp your objects are
    "brightness_min": 40,     # Minimum acceptable brightness
    "brightness_max": 210,    # Maximum acceptable brightness
    "color_var": 100.0        # Minimum color variance for richness
}

max_threads = 22
thread_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=max_threads)
object_predictions = {}  # track_id -> best-quality prediction (video mode)


def calculate_quality(frame, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return 0.0, False  # avoid crash

    # Convert to grayscale for blur & brightness
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)

    # Color variance
    color_var = np.var(crop)

    # Check thresholds
    quality_ok = (
        blur > QUALITY_THRESHOLDS["blur"] and
        QUALITY_THRESHOLDS["brightness_min"] <= brightness <= QUALITY_THRESHOLDS["brightness_max"] and
        color_var > QUALITY_THRESHOLDS["color_var"]
    )

    # Combine into one score if needed (optional)
    combined_score = (blur + brightness + color_var) / 3

    return combined_score, quality_ok
def ensure_output_dir(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        out_dir = input_path.parent / f"{input_path.stem}_output_yolo"
    else:
        out_dir = input_path / "output_yolo"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def process_single_image(img_path, output_dir):
    frame = cv2.imread(str(img_path))
    results = get_models().yolo.predict(frame, conf=0.5, iou=0.7)[0]  # Only detection, no tracking

    temp_id = 0

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            track_id = temp_id
            temp_id += 1

            try:
                quality, quality_ok = calculate_quality(frame, (x1, y1, x2, y2))
                crop = frame[y1:y2, x1:x2]

                if quality_ok:
                    pred_label = predict_with_resnet(crop)

                    if pred_label == "Healthy":
                        box_color = (0, 255, 0)
                    elif pred_label == "Marek":
                        box_color = (0, 165, 255)
                    elif pred_label == "Disease":
                        box_color = (0, 0, 255)
                    else:
                        box_color = (255, 255, 255)

                    neck_results = get_models().yolo_neck.predict(frame, conf=0.5, iou=0.5)[0]

                    if neck_results.boxes is not None and len(neck_results.boxes) > 0:
                        for neck_box in neck_results.boxes.xyxy:
                            nx1, ny1, nx2, ny2 = map(int, neck_box)
                            if nx1 >= x1 and ny1 >= y1 and nx2 <= x2 and ny2 <= y2:
                                try:
                                    neck_crop = frame[ny1:ny2, nx1:nx2]
                                    neck_quality, neck_quality_ok = calculate_quality(frame, (nx1, ny1, nx2, ny2))
                                    if neck_quality_ok:
                                        neck_pred = predict_with_resnet_neck(neck_crop)

                                        if neck_pred != "Healthy":
                                            neck_color = (0, 165, 255) if neck_pred == "infectious_coryza" else (0, 0, 255)
                                            cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), neck_color, 2)
                                            cv2.putText(frame, f"Neck: {neck_pred}", (nx1, ny2 + 20),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, neck_color, 2)
                                except Exception as e:
                                    print(f"❌ Neck classification error: {e}")
                else:
                    pred_label = "LowQuality"
                    box_color = (255, 255, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                label = f"{pred_label} (ID:{track_id})"
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                q_color = (0, 255, 255) if quality_ok else (0, 0, 255)
                cv2.putText(frame, f"Q:{quality:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, q_color, 2)

            except Exception as e:
                print(f"❌ Error in {Path(img_path).name}: {e}")

    else:
        print(f"🚫 No objects detected in image: {Path(img_path).name}")

    out_path = os.path.join(output_dir, f"{Path(img_path).stem}_classified.jpg")
    cv2.imwrite(out_path, frame)
    print(f"🖼️ Saved: {out_path}")
    return out_path  # <--- Return this path
def process_image_directory(input_dir, output_dir):
    image_files = [f for f in Path(input_dir).glob("*") if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    for img_path in image_files:
        process_single_image(img_path, output_dir)
def quality_and_classify(track_id, frame, box):
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]

    try:
        quality, quality_ok = calculate_quality(frame, box)

        if quality_ok:
            with thread_lock:
                last = object_predictions.get(track_id)
                if last is None or quality > last["quality"]:
                    prediction = predict_with_resnet(crop)
                    object_predictions[track_id] = {
                        "quality": quality,
                        "prediction": prediction
                    }


    except Exception as e:
        print(f"[Thread error] ID {track_id}: {e}")
def classify_neck_area(frame, full_body_box):
    try:
        neck_results = get_models().yolo_neck.predict(frame, conf=0.5, iou=0.5)[0]

        if neck_results.boxes is not None and len(neck_results.boxes) > 0:
            for neck_box in neck_results.boxes.xyxy:
                nx1, ny1, nx2, ny2 = map(int, neck_box)
                neck_crop = frame[ny1:ny2, nx1:nx2]

                # Check if neck box lies within full-body box
                fx1, fy1, fx2, fy2 = full_body_box
                if nx1 >= fx1 and ny1 >= fy1 and nx2 <= fx2 and ny2 <= fy2:
                    try:
                        neck_quality, neck_quality_ok = calculate_quality(frame, (nx1, ny1, nx2, ny2))
                        if neck_quality_ok:
                            neck_pred = predict_with_resnet_neck(neck_crop)

                            # Only draw if not healthy
                            if neck_pred != "Healthy":
                                neck_color = (0, 165, 255) if neck_pred == "infectious_coryza" else (0, 0, 255)
                                cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), neck_color, 2)
                                cv2.putText(frame, f"Neck: {neck_pred}", (nx1, ny2 + 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, neck_color, 2)
                    except Exception as e:
                        print(f"[Neck Detection Error] {e}")
    except Exception as e:
        print(f"[Neck Detection Outer Error] {e}")
def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(str(video_path))
    out_video_path = os.path.join(output_dir, f"{Path(video_path).stem}_tracked.mp4")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = get_models().yolo.track(frame, persist=True, conf=0.5, iou=0.5, tracker="botsort.yaml")[0]

        if results.boxes.id is not None and len(results.boxes.id) > 0:
            for box, track_id in zip(results.boxes.xyxy, results.boxes.id):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)

                # Thread for body classification
                executor.submit(quality_and_classify, track_id, frame.copy(), (x1, y1, x2, y2))

                # Thread for neck classification (only if body detected)
                executor.submit(classify_neck_area, frame.copy(), (x1, y1, x2, y2))

                # Draw green bounding box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw classification result if available
                prediction_info = object_predictions.get(track_id)
                if prediction_info:
                    pred_label = prediction_info["prediction"]
                    quality = prediction_info["quality"]

                    if pred_label == "Healthy":
                        pred_color = (0, 255, 0)
                    elif pred_label == "Marek":
                        pred_color = (0, 165, 255)
                    elif pred_label == "Disease":
                        pred_color = (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), pred_color, 2)
                    cv2.putText(frame, pred_label, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)
                    cv2.putText(frame, f"Q:{quality:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            print(f"[Frame {frame_index}] No objects detected.")

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    print(f"✅ Tracked video saved to: {out_video_path}")
def predict_with_resnet(crop):
    m = get_models()
    image = m.transform(crop).unsqueeze(0).to(m.device)  # [1, 3, 224, 224]
    with m.torch.no_grad():
        outputs = m.body(image)
        _, predicted = m.torch.max(outputs, 1)
        label = CLASS_LABELS[predicted.item()]
    return label

def predict_with_resnet_neck(crop):
    m = get_models()
    image = m.transform(crop).unsqueeze(0).to(m.device)
    with m.torch.no_grad():
        outputs = m.neck(image)
        _, predicted = m.torch.max(outputs, 1)
        label = CLASS_LABELS1[predicted.item()]
    return label
def run_detection_and_tracking(input_path, output_dir=None):
    input_path = Path(input_path)
    
    if output_dir is None:
        output_dir = ensure_output_dir(input_path)
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    if input_path.is_file() and input_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
        output_file = process_single_image(input_path, output_dir)

    elif input_path.is_file() and input_path.suffix.lower() in [".mp4", ".avi", ".mov"]:
        object_predictions.clear()
        yolo = get_models().yolo
        if hasattr(yolo, "predictor") and yolo.predictor is not None and hasattr(yolo.predictor, "tracker"):
            yolo.predictor.tracker = None

        output_file = process_video(input_path, output_dir)

    elif input_path.is_dir():
        output_file = process_image_directory(input_path, output_dir)

    else:
        print("Unsupported input format.")
        return None

    return output_file  # Return full output file path

