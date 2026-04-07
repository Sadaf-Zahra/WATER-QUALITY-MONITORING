import tkinter as tk
import serial
import serial.tools.list_ports
import threading
import numpy as np
import os
import time
import random
import json
import cv2
import csv
from datetime import datetime
from PIL import Image, ImageTk
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import joblib

import sys
import queue

# ===================== CONFIG SECTION =====================
# Static variables
BAUD_RATE = 9600
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 720
MODEL_FILE = "water_model_3sensor.pkl"
WASTE_MODEL_FILE = "simple_waste_model.h5"

CAMERA_INDEX = 0 #change here 1 for external cam use


# Sensor thresholds
TURBIDITY_LIMIT = 300
PH_MIN, PH_MAX = 6.5, 8.5
TDS_LIMIT = 500

# Default water sensor values if needed
DEFAULTS = {
    "Hardness": 200,
    "Solids": 20000,
    "Chloramines": 7,
    "Sulfate": 350,
    "Conductivity": 420,
    "Organic_carbon": 14,
    "Trihalomethanes": 70
}

# Waste labels and non-recyclable items
WASTE_LABELS = {
    0: 'plastic',
    1: 'cardboard',
    2: 'organic_waste',
    3: 'glass',
    4: 'metal',
    5: 'paper',
    6: 'cloth',
    7: 'wood',
    8: 'foam',
    9: 'others'
}

NON_RECYCLABLE = {"battery", "charger", "phone", "remote", "pendrive", "keys"}

# Directories
CSV_FILE = "logs/sensor_data.csv"
os.makedirs("captures", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# GUI settings
TITLE_FONT = ("Segoe UI", 20, "bold")
SECTION_FONT = ("Segoe UI", 14, "bold")
VALUE_FONT = ("Segoe UI", 16, "bold")
LABEL_FONT = ("Segoe UI", 12)
CARD_BG = "#ffffff"

# Model variables
water_model = None
waste_model = None
water_model_loaded = False
waste_model_loaded = False
last_predict_time = 0
last_waste_label = "NA"
last_waste_prob = 0.0

# Application state variables
is_fullscreen = False
camera_running = True
latest_frame = None

# Queue for thread-safe GUI updates
gui_queue = queue.Queue()

# ===================== FUNCTION DEFINITIONS =====================

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        print("relative_path:", os.path.join(sys._MEIPASS, relative_path))
        return os.path.join(sys._MEIPASS, relative_path)
    print("relative_path:", os.path.join(os.path.abspath("."), relative_path))
    return os.path.join(os.path.abspath("."), relative_path)

def process_gui_queue():
    """Process all pending GUI updates from the queue"""
    try:
        while True:
            try:
                task = gui_queue.get_nowait()
                if callable(task):
                    task()
            except queue.Empty:
                break
    except Exception as e:
        print(f"Error processing GUI queue: {e}")
    
    # Schedule the next check
    if root.winfo_exists():
        root.after(100, process_gui_queue)

def schedule_gui_update(task):
    """Safely schedule a GUI update from any thread"""
    gui_queue.put(task)

def toggle_fullscreen(event=None):
    global is_fullscreen
    is_fullscreen = not is_fullscreen
    root.attributes("-fullscreen", is_fullscreen)

def exit_fullscreen(event=None):
    global is_fullscreen
    is_fullscreen = False
    root.attributes("-fullscreen", False)

def load_waste_model():
    global waste_model, waste_model_loaded
    try:
        waste_model = tf.keras.models.load_model(
            resource_path(WASTE_MODEL_FILE)
        )
        waste_model_loaded = True
        print("[INFO] Waste model loaded successfully")
    except Exception as e:
        waste_model_loaded = False
        print("[ERROR] Waste model load failed:", e)


def preprocess_waste(img):
    img = img.resize((224, 224))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    return img_array


def waste_predict(frame):
    if not waste_model_loaded:
        return "NA", 0.0

    img = Image.fromarray(frame)
    img = preprocess_waste(img)

    prediction = waste_model.predict(
        img[np.newaxis, ...], verbose=0
    )

    prob = float(np.max(prediction[0]))
    class_index = int(np.argmax(prediction[0]))

    label = WASTE_LABELS.get(class_index, "Unknown")
    return label, round(prob, 2)


def model_decision(ph, tds, turb):
    if not water_model_loaded:
        return None, None

    sample = pd.DataFrame([{
        "ph": ph,
        "Solids": tds,
        "Turbidity": turb
    }])

    prob = water_model.predict_proba(sample)[0][1]
    return prob > 0.5, prob

def set_camera(state):
    global camera_running
    camera_running = state

def capture_image():
    if latest_frame is not None:
        cv2.imwrite(
            f"captures/capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            latest_frame
        )

def set_connection_status_safe(text):
    # Schedule the update through the queue
    def update():
        if root.winfo_exists():
            connection_status.set(text)
    schedule_gui_update(update)

def update_camera():
    global latest_frame, last_predict_time
    global last_waste_label, last_waste_prob

    if camera_running:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame.copy()
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            lw, lh = camera_label.winfo_width(), camera_label.winfo_height()
            if lw > 1 and lh > 1:
                h, w, _ = frame.shape
                scale = min(lw / w, lh / h)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            img = ImageTk.PhotoImage(Image.fromarray(frame))
            camera_label.configure(image=img)
            camera_label.image = img

            if waste_model_loaded and time.time() - last_predict_time > 1.0:
                last_waste_label, last_waste_prob = waste_predict(latest_frame)
                last_predict_time = time.time()

            if last_waste_prob >= 0.4:
                def update_waste():
                    waste_status.set("WASTE DETECTED")
                    waste_name.set(f"{last_waste_label} ({last_waste_prob})")
                    waste_label.config(
                        fg="red" if last_waste_label in NON_RECYCLABLE else "green"
                    )
                schedule_gui_update(update_waste)
            else:
                def update_no_waste():
                    waste_status.set("NO WASTE DETECTED")
                    waste_name.set("-------")
                schedule_gui_update(update_no_waste)

    camera_label.after(30, update_camera)

def find_arduino_port():
    for port in serial.tools.list_ports.comports():
        if any(x in port.description.lower() for x in ["arduino", "ch340", "usb serial"]):
            return port.device
    return None

def connect_serial():
    while True:
        try:
            port = find_arduino_port()
            if port:
                ser = serial.Serial(port, BAUD_RATE, timeout=1)
                set_connection_status_safe(f"CONNECTED ({port}) ✅")
                time.sleep(2)
                return ser
            else:
                set_connection_status_safe("WAITING FOR ARDUINO ⏳")
        except:
            set_connection_status_safe("RECONNECTING ❌")
        time.sleep(2)


def save_to_csv(time_str, ph, tds, turb, prediction="NA", probability="NA"):
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Time", "pH", "TDS", "Turbidity",
                "Prediction", "Probability"
            ])

        writer.writerow([
            time_str,
            ph,
            tds,
            turb,
            prediction,
            probability
        ])

def update_sensor_ui(ph, tds, turb, safe):
    ph_var.set(f"{ph:.2f}")
    tds_var.set(f"{tds:.0f}")
    turb_var.set(f"{turb:.0f}")

    last_update_var.set(
        f"Last update: {datetime.now():%Y-%m-%d %H:%M:%S}"
    )

    sensor_status.set("SAFE ✅" if safe else "UNSAFE ❌")
    sensor_status_label.config(
        fg="green" if safe else "red"
    )

    final_status.set("DRINKABLE ✅" if safe else "NOT DRINKABLE ❌")
    final_label.config(fg="green" if safe else "red")

def update_model_ui(prediction, prob):
    if prediction == "NA":
        model_status.set("MODEL NOT LOADED")
        model_status_label.config(fg="gray")
    elif prediction == "SAFE":
        model_status.set(f"SAFE ({prob})")
        model_status_label.config(fg="green")
    else:  # UNSAFE
        model_status.set(f"UNSAFE ({prob})")
        model_status_label.config(fg="red")

def start_serial_thread():
    threading.Thread(target=read_serial, daemon=True).start()

def read_serial():
    ser = connect_serial()

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if not line or not line.startswith("{"):
                continue

            data = json.loads(line)

            ph = float(data.get("ph", 0))
            tds = float(data.get("tds", 0))
            turb = float(data.get("turbidity", 0))

            # ---------- SENSOR DECISION ----------
            sensor_safe = (
                PH_MIN <= ph <= PH_MAX
                and tds < TDS_LIMIT
                and turb < TURBIDITY_LIMIT
            )

            schedule_gui_update(lambda: update_sensor_ui(ph, tds, turb, sensor_safe))

            # ---------- DEFAULT MODEL VALUES ----------
            prediction = "NA"
            prob_val = "NA"

            # ---------- MODEL DECISION ----------
            if water_model_loaded:
                try:
                    m_ok, prob = model_decision(ph, tds, turb)

                    if m_ok is not None and prob is not None:
                        prediction = "SAFE" if m_ok else "UNSAFE"
                        prob_val = round(prob, 2)

                        print(
                            f"Model prediction: {prediction}, "
                            f"probability: {prob_val}"
                        )

                except Exception as e:
                    print("Model inference failed:", e)
                    # prediction & prob_val stay as "NA"

            # ---------- MODEL UI UPDATE ----------
            schedule_gui_update(lambda: update_model_ui(prediction, prob_val))

            # ---------- CSV SAVE (ALWAYS) ----------
            if auto_save_csv.get():
                save_to_csv(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ph, tds, turb,
                    prediction, prob_val
                )

        except Exception as e:
            print(f"Serial read error: {e}")
            try:
                ser.close()
            except:
                pass

            set_connection_status_safe("ARDUINO DISCONNECTED ❌")

            time.sleep(2)
            ser = connect_serial()
def open_sensor_help():
    help_win = tk.Toplevel(root)
    help_win.title("Sensor Limits & Live Values")
    help_win.geometry("650x520")
    help_win.configure(bg="#f9fafb")
    help_win.resizable(False, False)

    # ================= Title =================
    tk.Label(
        help_win,
        text="Sensor Limits & Live Readings",
        font=("Segoe UI", 16, "bold"),
        bg="#f9fafb"
    ).pack(pady=8)

    # ================= Live Values =================
    live_frame = tk.Frame(help_win, bg="#eef2f7", bd=1, relief="solid")
    live_frame.pack(fill="x", padx=15, pady=8)

    tk.Label(
        live_frame,
        text="Live Sensor Values",
        font=("Segoe UI", 13, "bold"),
        bg="#eef2f7"
    ).pack(anchor="w", padx=10, pady=5)

    row = tk.Frame(live_frame, bg="#eef2f7")
    row.pack(padx=10, pady=6)

    # pH
    tk.Label(row, text="pH:", font=LABEL_FONT, bg="#eef2f7") \
        .grid(row=0, column=0, sticky="w")
    tk.Label(row, textvariable=ph_var, font=VALUE_FONT, bg="#eef2f7") \
        .grid(row=0, column=1)
    tk.Label(row, text="(pH)", font=LABEL_FONT, bg="#eef2f7") \
        .grid(row=0, column=2, padx=(0, 15))

    # TDS
    tk.Label(row, text="TDS:", font=LABEL_FONT, bg="#eef2f7") \
        .grid(row=0, column=3, sticky="w")
    tk.Label(row, textvariable=tds_var, font=VALUE_FONT, bg="#eef2f7") \
        .grid(row=0, column=4)
    tk.Label(row, text="(ppm)", font=LABEL_FONT, bg="#eef2f7") \
        .grid(row=0, column=5, padx=(0, 15))

    # Turbidity
    tk.Label(row, text="Turbidity:", font=LABEL_FONT, bg="#eef2f7") \
        .grid(row=0, column=6, sticky="w")
    tk.Label(row, textvariable=turb_var, font=VALUE_FONT, bg="#eef2f7") \
        .grid(row=0, column=7)
    tk.Label(row, text="(NTU)", font=LABEL_FONT, bg="#eef2f7") \
        .grid(row=0, column=8)

    # ================= Scrollable Description =================
    desc_frame = tk.Frame(help_win)
    desc_frame.pack(fill="both", expand=True, padx=15, pady=10)

    scrollbar = tk.Scrollbar(desc_frame)
    scrollbar.pack(side="right", fill="y")

    text = tk.Text(
        desc_frame,
        wrap="word",
        yscrollcommand=scrollbar.set,
        font=("Segoe UI", 11),
        bg="white",
        relief="solid",
        bd=1
    )
    text.pack(fill="both", expand=True)
    scrollbar.config(command=text.yview)

    description = f"""
🧪 pH SENSOR
Measurement Unit: pH (unitless)
Safe Range: {PH_MIN} – {PH_MAX} pH
Description:
Measures acidity or alkalinity of water.
Low pH (<6.5) causes corrosion risk.
High pH (>8.5) causes bitter taste & scaling.

💧 TDS SENSOR
Measurement Unit: ppm (mg/L)
Safe Limit: < {TDS_LIMIT} ppm
Description:
Measures total dissolved solids.
Includes salts, minerals, and metals.
High TDS affects taste and long-term health.

🌫 TURBIDITY SENSOR
Measurement Unit: NTU
Safe Limit: < {TURBIDITY_LIMIT} NTU
Description:
Measures water clarity.
High turbidity indicates:
• Mud & silt
• Microorganisms
• Organic contamination

✅ WATER SAFETY DECISION LOGIC
Water is marked SAFE only if:
• pH is between {PH_MIN} – {PH_MAX} pH
• TDS is below {TDS_LIMIT} ppm
• Turbidity is below {TURBIDITY_LIMIT} NTU

❌ If ANY sensor exceeds its limit → Water is NOT DRINKABLE
"""

    text.insert("1.0", description)
    text.config(state="disabled")  # Read-only

    # ================= Close Button =================
    tk.Button(
        help_win,
        text="Close",
        font=("Segoe UI", 11, "bold"),
        command=help_win.destroy
    ).pack(pady=8)



# ===================== GUI SETUP =====================
# Initialize root window
root = tk.Tk()
root.bind("<F11>", toggle_fullscreen)
root.bind("<Escape>", exit_fullscreen)
root.title("Water Quality & Waste Monitoring System")

# Center window
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (WINDOW_WIDTH // 2)
y = (screen_height // 2) - (WINDOW_HEIGHT // 2)
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")
root.configure(bg="#f2f4f7")
root.resizable(False, False)

# Start GUI queue processing
root.after(100, process_gui_queue)

# Load water model
try:
    water_model = joblib.load(resource_path(MODEL_FILE))
    water_model_loaded = True
    isloaded_model = tk.StringVar(value="WATER MODEL LOADED ✅")
except Exception as e:
    water_model_loaded = False
    isloaded_model = tk.StringVar(value="WATER MODEL NOT LOADED ❌")
    print("Water model load error:", e)

# Tkinter variables
ph_var = tk.StringVar(value="--")
tds_var = tk.StringVar(value="--")
turb_var = tk.StringVar(value="--")
last_update_var = tk.StringVar(value="Last update: --")
connection_status = tk.StringVar(value="WAITING FOR ARDUINO")
sensor_status = tk.StringVar(value="WAITING")
model_status = tk.StringVar(value="NA")
final_status = tk.StringVar(value="WAITING")
waste_status = tk.StringVar(value="NO WASTE DETECTED")
waste_name = tk.StringVar(value="-------")
auto_save_csv = tk.BooleanVar(value=False)

# Title
tk.Label(root, text="Water Quality & Waste Monitoring",
         font=TITLE_FONT, bg="#f2f4f7").pack(pady=10)

# Main layout
main = tk.Frame(root, bg="#f2f4f7")
main.pack(fill="both", expand=True, padx=10)

left = tk.Frame(main, bg="#f2f4f7")
left.pack(side="left", fill="y", padx=5)

right = tk.Frame(main, bg="#f2f4f7")
right.pack(side="right", fill="both", expand=True, padx=5)

# Sensor card
sensor_card = tk.Frame(left, bg=CARD_BG, bd=1, relief="solid")
sensor_card.pack(fill="x", pady=8)

header_frame = tk.Frame(sensor_card, bg=CARD_BG)
header_frame.pack(fill="x", padx=15, pady=10)

tk.Label(
    header_frame,
    text="Sensor Readings",
    font=SECTION_FONT,
    bg=CARD_BG
).pack(side="left")

tk.Button(
    header_frame,
    text="ℹ️",
    font=("Segoe UI", 12, "bold"),
    bg=CARD_BG,
    bd=0,
    cursor="hand2",
    command=open_sensor_help
).pack(side="left", padx=6)



grid = tk.Frame(sensor_card, bg=CARD_BG)
grid.pack(padx=15, pady=10)

tk.Label(grid, text="pH", font=LABEL_FONT, bg=CARD_BG).grid(row=0, column=0)
tk.Label(grid, textvariable=ph_var, font=VALUE_FONT, bg=CARD_BG).grid(row=1, column=0, padx=15)

tk.Label(grid, text="TDS", font=LABEL_FONT, bg=CARD_BG).grid(row=0, column=1)
tk.Label(grid, textvariable=tds_var, font=VALUE_FONT, bg=CARD_BG).grid(row=1, column=1, padx=15)

tk.Label(grid, text="Turbidity", font=LABEL_FONT, bg=CARD_BG).grid(row=0, column=2)
tk.Label(grid, textvariable=turb_var, font=VALUE_FONT, bg=CARD_BG).grid(row=1, column=2, padx=15)
tk.Label(
    sensor_card,
    textvariable=last_update_var,
    font=("Segoe UI", 10),
    bg=CARD_BG,
    fg="gray"
).pack(anchor="e", padx=15, pady=(0, 5))

# Decision card
decision_card = tk.Frame(left, bg=CARD_BG, bd=1, relief="solid")
decision_card.pack(fill="x", pady=8)

tk.Label(decision_card, text="Decision Status",
         font=SECTION_FONT, bg=CARD_BG).pack(anchor="w", padx=15, pady=10)

tk.Label(decision_card, text="Sensor Decision", font=LABEL_FONT, bg=CARD_BG)\
    .pack(anchor="w", padx=15)

sensor_status_label = tk.Label(
    decision_card,
    textvariable=sensor_status,
    font=VALUE_FONT,
    bg=CARD_BG
)
sensor_status_label.pack(anchor="w", padx=15)

tk.Label(decision_card, text="Model Decision", font=LABEL_FONT, bg=CARD_BG)\
    .pack(anchor="w", padx=15, pady=(10, 0))

model_status_label = tk.Label(
    decision_card,
    textvariable=model_status,
    font=VALUE_FONT,
    bg=CARD_BG
)
model_status_label.pack(anchor="w", padx=15)

# Final status card
final_card = tk.Frame(left, bg=CARD_BG, bd=2, relief="solid")
final_card.pack(fill="x", pady=10)

tk.Label(final_card, text="Final Water Status",
         font=SECTION_FONT, bg=CARD_BG).pack(pady=8)

final_label = tk.Label(final_card, textvariable=final_status,
                       font=("Segoe UI", 22, "bold"), bg=CARD_BG)
final_label.pack(pady=10)

# Waste card
waste_card = tk.Frame(left, bg=CARD_BG, bd=1, relief="solid")
waste_card.pack(fill="x", pady=8)

tk.Label(waste_card, text="Waste Monitoring",
         font=SECTION_FONT, bg=CARD_BG).pack(anchor="w", padx=15, pady=10)

waste_label = tk.Label(waste_card, textvariable=waste_status,
                       font=("Segoe UI", 16, "bold"),
                       bg=CARD_BG, fg="green")
waste_label.pack(anchor="w", padx=15)

tk.Label(waste_card, textvariable=waste_name,
         font=LABEL_FONT, bg=CARD_BG)\
    .pack(anchor="w", padx=15, pady=(0, 10))

# Camera panel
camera_card = tk.Frame(right, bg=CARD_BG, bd=1, relief="solid")
camera_card.pack(fill="both", expand=True, pady=8)

# Camera header
camera_header = tk.Frame(camera_card, bg=CARD_BG)
camera_header.pack(fill="x", padx=15, pady=10)

tk.Label(
    camera_header,
    text="Live Camera Feed",
    font=SECTION_FONT,
    bg=CARD_BG
).pack(side="left")

# Status box
status_box = tk.Frame(camera_header, bg=CARD_BG)
status_box.pack(side="right")

# Connection Status
tk.Label(
    status_box,
    text="Connection:",
    font=("Segoe UI", 10),
    bg=CARD_BG,
    fg="gray"
).grid(row=0, column=0, sticky="e", padx=(0, 4))

tk.Label(
    status_box,
    textvariable=connection_status,
    font=("Segoe UI", 10, "bold"),
    bg=CARD_BG,
    fg="green"
).grid(row=0, column=1, sticky="w", padx=(0, 15))

# Model Loaded Status
tk.Label(
    status_box,
    text="Model:",
    font=("Segoe UI", 10),
    bg=CARD_BG,
    fg="gray"
).grid(row=0, column=2, sticky="e", padx=(0, 4))

tk.Label(
    status_box,
    textvariable=isloaded_model,
    font=("Segoe UI", 10, "bold"),
    bg=CARD_BG,
    fg="blue"
).grid(row=0, column=3, sticky="w")

# Controls
controls = tk.Frame(camera_card, bg=CARD_BG)
controls.pack(fill="x", padx=15, pady=(0, 10))

tk.Button(controls, text="▶ PLAY", bg="#2ecc71", fg="white",
          font=("Segoe UI", 11, "bold"),
          command=lambda: set_camera(True)).pack(side="left", padx=5)

tk.Button(controls, text="⏸ PAUSE", bg="#e74c3c", fg="white",
          font=("Segoe UI", 11, "bold"),
          command=lambda: set_camera(False)).pack(side="left", padx=5)

tk.Button(controls, text="📸 Capture", font=("Segoe UI", 11),
    command=capture_image
).pack(side="left", padx=10)

tk.Checkbutton(controls, text="💾 Auto-save CSV",
               variable=auto_save_csv, bg=CARD_BG)\
    .pack(side="right")

# Camera view
camera_view = tk.Frame(camera_card, bg="black")
camera_view.pack(fill="both", expand=True, padx=15, pady=10)

camera_label = tk.Label(camera_view, bg="black")
camera_label.pack(fill="both", expand=True)

# ===================== INITIALIZATION =====================
# Initialize camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("[ERROR] Camera not accessible")

# Start threads
start_serial_thread()
load_waste_model()
update_camera()

# Start main loop
root.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()