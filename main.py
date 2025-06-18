import cv2
import csv
import datetime
import serial
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

try:
    ser = serial.Serial('COM5', 115200, timeout=1)
    print("✅ Serial ESP32 terhubung.")
except Exception as e:
    print("❌ Gagal hubungkan ESP32:", e)
    ser = None

def kirim_ke_esp32(cx, width=640):
    tengah = width // 2
    toleransi = 60

    if ser and ser.is_open:
        if cx < tengah - toleransi:
            ser.write(b'3')  # Belok kiri
        elif cx > tengah + toleransi:
            ser.write(b'2')  # Belok kanan
        else:
            ser.write(b'1')  # Maju lurus

print("📦 Memuat model YOLO...")
model = YOLO("yolov8n.pt")
print("✅ Model YOLO berhasil dimuat.")

tracker = DeepSort(max_age=30)
locked_target_id = None
last_boxes = {}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Gagal membuka kamera.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))
log_file = open("log_target.csv", mode="w", newline='')
log_writer = csv.writer(log_file)
log_writer.writerow(["Timestamp", "Target_ID", "Center_X", "Center_Y"])

def mouse_callback(event, x, y, flags, param):
    global locked_target_id
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"🖱️ Klik di ({x}, {y})")
        for track_id, (hx1, hy1, hx2, hy2) in last_boxes.items():
            if hx1 <= x <= hx2 and hy1 <= y <= hy2:
                locked_target_id = track_id
                print(f"🔒 Target dikunci: ID {track_id}")
                break

cv2.namedWindow("🔴 Tracking Manusia")
cv2.setMouseCallback("🔴 Tracking Manusia", mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []
    id_to_head = {}

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 0 and conf > 0.4: 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1

            head_h = int(h * 0.4)
            cx = x1 + w // 2
            hx1 = max(0, cx - w // 2)
            hy1 = y1
            hx2 = min(frame.shape[1], cx + w // 2)
            hy2 = y1 + head_h

            detections.append(([hx1, hy1, hx2 - hx1, hy2 - hy1], conf, "person"))
            id_to_head[(hx1, hy1, hx2, hy2)] = None

    tracks = tracker.update_tracks(detections, frame=frame)
    last_boxes.clear()

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        w, h = r - l, b - t
        cx, cy = l + w // 2, t + h // 4

        matched_head = None
        for (hx1, hy1, hx2, hy2) in id_to_head:
            if hx1 <= cx <= hx2 and hy1 <= cy <= hy2:
                matched_head = (hx1, hy1, hx2, hy2)
                break

        if matched_head:
            last_boxes[track_id] = matched_head

        if track_id == locked_target_id:
            color = (0, 255, 0)
            label = f"🎯 LOCKED - ID {track_id}"
            cv2.circle(frame, (cx, cy), 6, color, -1)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), color, 2)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), color, 2)

            kirim_ke_esp32(cx, frame.shape[1])

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_writer.writerow([timestamp, track_id, cx, cy])
        else:
            color = (100, 100, 100)
            label = f"ID {track_id}"

        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, label, (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for (hx1, hy1, hx2, hy2) in last_boxes.values():
        center_x = (hx1 + hx2) // 2
        center_y = (hy1 + hy2) // 2
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
        cv2.circle(frame, (center_x, center_y), 4, (255, 0, 255), -1)

    if locked_target_id is not None:
        cv2.putText(frame, f"🎯 Target LOCKED: ID {locked_target_id}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("🔴 Tracking Manusia", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("👋 Keluar...")
        break
    elif key == ord('u') and locked_target_id is not None:
        print(f"🔓 Unlock ID {locked_target_id}")
        locked_target_id = None
        if ser:
            ser.write(b'0')  # Stop jika unlock
    elif key == ord('l') and len(last_boxes) > 0:
        center_frame_x = frame.shape[1] // 2
        center_frame_y = frame.shape[0] // 4
        min_dist = float('inf')
        chosen_id = None
        for tid, (hx1, hy1, hx2, hy2) in last_boxes.items():
            cx = (hx1 + hx2) // 2
            cy = (hy1 + hy2) // 2
            dist = (center_frame_x - cx) ** 2 + (center_frame_y - cy) ** 2
            if dist < min_dist:
                min_dist = dist
                chosen_id = tid
        if chosen_id:
            locked_target_id = chosen_id
            print(f"🔒 LOCK otomatis ke ID {chosen_id}")

cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()

if ser:
    ser.write(b'0')  # Pastikan berhenti saat keluar
    ser.close()
