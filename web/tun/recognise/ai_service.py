import base64
import logging
import os
import threading
import cv2
import numpy as np
from PIL import Image
import io
from django.conf import settings

logger = logging.getLogger(__name__)

# Emotions recognized by the CNN model
EMOTIONS = ["afraid", "angry", "disgust", "happy", "neutral", "sad", "surprised"]
IMG_SIZE = 48

# Singleton model & cascade containers with thread lock
_model_lock = threading.Lock()
_emotion_model = None
_face_cascade = None


def get_emotion_model():
    """
    Lazy thread-safe loader for the CNN emotion classifier.
    Uses tf_keras to maintain compatibility with Keras 2 SavedModel format.
    """
    global _emotion_model
    if _emotion_model is None:
        with _model_lock:
            if _emotion_model is None:
                import tf_keras
                model_path = os.path.join(settings.BASE_DIR, '6')
                if not os.path.exists(model_path):
                    # Check parent repository CNN-model as fallback
                    alt_path = os.path.join(settings.BASE_DIR.parent.parent, 'CNN-model')
                    if os.path.exists(alt_path):
                        model_path = alt_path
                logger.info(f"Loading emotion CNN model from: {model_path}")
                _emotion_model = tf_keras.models.load_model(model_path, compile=False)
                logger.info("Emotion CNN model loaded successfully.")
    return _emotion_model


def get_face_cascade():
    """
    Lazy thread-safe loader for Haar Cascade frontal face detector.
    """
    global _face_cascade
    if _face_cascade is None:
        with _model_lock:
            if _face_cascade is None:
                xml_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
                if not os.path.exists(xml_path):
                    xml_path = os.path.join(settings.BASE_DIR, 'haarcascade_frontalface_default.xml')
                _face_cascade = cv2.CascadeClassifier(xml_path)
                if _face_cascade.empty():
                    logger.error(f"Failed to load Haar Cascade from {xml_path}")
    return _face_cascade


def decode_image(image_input):
    """
    Decodes various image inputs (base64 data URL, bytes, PIL.Image, numpy.ndarray) into a BGR OpenCV image.
    """
    if isinstance(image_input, np.ndarray):
        return image_input.copy()

    if isinstance(image_input, Image.Image):
        rgb_image = np.array(image_input)
        if len(rgb_image.shape) == 2:
            return cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    if isinstance(image_input, str):
        if image_input.startswith('data:image'):
            # Strip data URL header e.g. "data:image/jpeg;base64,"
            _, encoded = image_input.split(',', 1)
            image_bytes = base64.b64decode(encoded)
        else:
            image_bytes = base64.b64decode(image_input)
    elif hasattr(image_input, 'read'):
        image_bytes = image_input.read()
    elif isinstance(image_input, (bytes, bytearray)):
        image_bytes = bytes(image_input)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image bytes into OpenCV image")
    return img_bgr


def encode_image_to_base64(img_bgr, quality=85):
    """
    Encodes OpenCV BGR image to base64 JPEG data URL.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode('.jpg', img_bgr, encode_param)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"


def preprocess_roi(roi_bgr):
    """
    Converts ROI to grayscale, normalizes to [0, 1], and resizes to 48x48.
    Returns tensor shape: (1, 48, 48, 1).
    """
    if len(roi_bgr.shape) == 3 and roi_bgr.shape[2] == 3:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_bgr.copy()
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def analyze_emotion(image_input):
    """
    Full pipeline: decodes image, detects face, classifies emotion with CNN model,
    draws bounding boxes, and returns complete analysis.
    """
    img_bgr = decode_image(image_input)
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    cascade = get_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    face_detected = len(faces) > 0
    face_box = None
    annotated_img = img_bgr.copy()

    if face_detected:
        # Choose the largest detected face
        faces = sorted(faces, reverse=True, key=lambda f: f[2] * f[3])
        x, y, fw, fh = faces[0]
        face_box = [int(x), int(y), int(fw), int(fh)]
        roi = img_bgr[y:y+fh, x:x+fw]
    else:
        # Center square crop fallback
        side = min(h, w)
        sy = (h - side) // 2
        sx = (w - side) // 2
        roi = img_bgr[sy:sy+side, sx:sx+side]

    # Preprocess & Predict
    tensor_input = preprocess_roi(roi)
    model = get_emotion_model()
    raw_preds = model.predict(tensor_input, verbose=0)[0]

    # Normalize to probabilities
    exp_preds = np.exp(raw_preds - np.max(raw_preds))
    probs = exp_preds / np.sum(exp_preds)

    max_idx = int(np.argmax(probs))
    primary_emotion = EMOTIONS[max_idx]
    confidence = float(probs[max_idx])

    prob_dict = {EMOTIONS[i]: round(float(probs[i]), 4) for i in range(len(EMOTIONS))}

    # Annotate image
    if face_detected:
        x, y, fw, fh = face_box
        # Modern rounded-style bounding box
        cv2.rectangle(annotated_img, (x, y), (x + fw, y + fh), (29, 185, 84), 2) # Spotify green
        # Header banner with emotion
        label_text = f"{primary_emotion.upper()} ({confidence*100:.1f}%)"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        bg_y1 = max(0, y - th - 12)
        cv2.rectangle(annotated_img, (x, bg_y1), (x + tw + 16, y), (29, 185, 84), -1)
        cv2.putText(annotated_img, label_text, (x + 8, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    annotated_b64 = encode_image_to_base64(annotated_img)

    return {
        "success": True,
        "face_detected": face_detected,
        "emotion": primary_emotion,
        "confidence": confidence,
        "probabilities": prob_dict,
        "face_box": face_box,
        "annotated_image": annotated_b64,
    }
