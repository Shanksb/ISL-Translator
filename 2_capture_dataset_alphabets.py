import cv2
import mediapipe as mp
from pathlib import Path
import time
import string

BASE_DIR = Path("dataset_raw_alphabets")
BASE_DIR.mkdir(exist_ok=True)

LABEL_KEYS = {ord(ch.lower()): ch for ch in string.ascii_uppercase}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                    min_detection_confidence=0.5) as hands:
    record_label = None
    img_count = {}

    print("Controls")
    print("Press any key a–z → start recording for that alphabet")
    print("ESC → quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        display = frame.copy()
        h, w, _ = frame.shape

        # Draw hand marks
        if res.multi_hand_landmarks:
            for hm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(display, hm, mp_hands.HAND_CONNECTIONS)

            # Draw rectangle around hands
            all_x, all_y = [], []
            for hm in res.multi_hand_landmarks:
                all_x.extend([lm.x for lm in hm.landmark])
                all_y.extend([lm.y for lm in hm.landmark])
            x1, x2 = int(min(all_x) * w), int(max(all_x) * w)
            y1, y2 = int(min(all_y) * h), int(max(all_y) * h)
            cv2.rectangle(display, (x1 - 20, y1 - 20),
                          (x2 + 20, y2 + 20), (0, 255, 0), 2)

        # Show current label on screen
        cv2.putText(display, f"Recording: {record_label or 'None'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Capture Dataset (A–Z)", display)
        key = cv2.waitKey(1) & 0xFF

        # Start recording for an alphabet
        if key in LABEL_KEYS:
            record_label = LABEL_KEYS[key]
            folder = BASE_DIR / record_label
            folder.mkdir(parents=True, exist_ok=True)
            img_count.setdefault(record_label, 0)
            print(f"Started recording label: {record_label}")

        elif key == ord('s'):
            record_label = None
            print("Stopped recording.")

        elif key in (ord('q'), 27):
            break
        # Save images
        if record_label and res.multi_hand_landmarks:
            all_x, all_y = [], []
            for hm in res.multi_hand_landmarks:
                all_x.extend([lm.x for lm in hm.landmark])
                all_y.extend([lm.y for lm in hm.landmark])

            x1, x2 = int(min(all_x) * w), int(max(all_x) * w)
            y1, y2 = int(min(all_y) * h), int(max(all_y) * h)

            crop = frame[max(0, y1 - 20):min(h, y2 + 20),
                         max(0, x1 - 20):min(w, x2 + 20)]

            if crop.size != 0:
                crop = cv2.resize(crop, (224, 224))
                fname = BASE_DIR / record_label / f"{record_label}_{img_count[record_label]:04d}.jpg"
                cv2.imwrite(str(fname), crop)
                img_count[record_label] += 1
                time.sleep(0.05) #time delay for saving images

cap.release()
cv2.destroyAllWindows()
