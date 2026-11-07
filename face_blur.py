"""
Face Blur App — OpenCV mini project
-----------------------------------
Detects faces from a webcam or video file and blurs, pixelates, or boxes them.

Usage:
    python face_blur.py              # webcam
    python face_blur.py video.mp4    # video file

Keys:
    b = blur mode      p = pixelate mode    d = box mode
    + = stronger       - = weaker
    s = screenshot     r = start/stop recording
    q = quit
"""

import os
import sys
import time
from datetime import datetime

import cv2

# ---------- Settings ----------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use a video file if one is given on the command line, otherwise the webcam
SOURCE = sys.argv[1] if len(sys.argv) > 1 else 0
IS_WEBCAM = SOURCE == 0

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------- Helper functions ----------
def blur_face(roi, strength):
    """Blur a face region. Kernel size scales with the face and is always odd."""
    h, w = roi.shape[:2]
    k = max(w // strength, 1) | 1          # "| 1" forces an odd number
    return cv2.GaussianBlur(roi, (k, k), 0)


def pixelate_face(roi, blocks):
    """Shrink to a few pixels, then enlarge with NEAREST for a blocky look."""
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def draw_box(img, x1, y1, x2, y2, label, color=(0, 255, 0)):
    """Draw a bounding box with a filled label background."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------- Main program ----------
def main():
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print("Could not open the camera or video file!")
        return

    # Recording FPS: use the file's FPS, or 20 for webcams
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    record_fps = src_fps if src_fps and src_fps > 0 and not IS_WEBCAM else 20

    mode = "blur"          # blur | pixel | box
    strength = 3           # blur: smaller = stronger (face width / strength)
    blocks = 10            # pixelate: fewer blocks = more blocky
    writer = None          # active VideoWriter while recording
    prev_time = time.time()

    print(__doc__)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if IS_WEBCAM:
            frame = cv2.flip(frame, 1)     # mirror view for webcams

        # 1. Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        # 2. Apply the current mode to each face
        for i, (x, y, w, h) in enumerate(faces, start=1):
            roi = frame[y:y + h, x:x + w]
            if mode == "blur":
                frame[y:y + h, x:x + w] = blur_face(roi, strength)
            elif mode == "pixel":
                frame[y:y + h, x:x + w] = pixelate_face(roi, blocks)
            else:
                draw_box(frame, x, y, x + w, y + h, f"Face {i}")

        # 3. Record BEFORE drawing the info overlay, so the video stays clean
        if writer is not None:
            writer.write(frame)

        # 4. On-screen info
        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now

        display = frame.copy()
        level = strength if mode == "blur" else blocks
        info = f"Mode: {mode} | Level: {level} | Faces: {len(faces)} | FPS: {fps:.1f}"
        cv2.putText(display, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        if writer is not None:
            cv2.circle(display, (display.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display, "REC", (display.shape[1] - 80, 37),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Face Blur App", display)

        # 5. Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("b"):
            mode = "blur"
        elif key == ord("p"):
            mode = "pixel"
        elif key == ord("d"):
            mode = "box"
        elif key in (ord("+"), ord("=")):
            if mode == "blur":
                strength = max(1, strength - 1)      # smaller divisor = bigger kernel
            else:
                blocks = max(3, blocks - 1)
        elif key == ord("-"):
            if mode == "blur":
                strength = min(10, strength + 1)
            else:
                blocks = min(40, blocks + 1)
        elif key == ord("s"):
            path = os.path.join(OUTPUT_DIR, f"shot_{timestamp()}.jpg")
            cv2.imwrite(path, frame)
            print("Screenshot saved:", path)
        elif key == ord("r"):
            if writer is None:
                h, w = frame.shape[:2]
                path = os.path.join(OUTPUT_DIR, f"rec_{timestamp()}.mp4")
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         record_fps, (w, h))
                print("Recording started:", path)
            else:
                writer.release()
                writer = None
                print("Recording stopped")

    # ---------- Cleanup ----------
    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
