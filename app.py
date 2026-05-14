import streamlit as st
import cv2
import cvzone
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(
    page_title="VisionSafe | PPE AI",
    page_icon="🛡️",
    layout="wide"
)

# 2. Advanced Custom CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            font-family: 'Inter', sans-serif;
        }

        .header-box {
            background: linear-gradient(90deg, #FF4B2B 0%, #FF8008 100%);
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(255, 75, 43, 0.3);
        }

        [data-testid="stSidebar"] {
            background-color: rgba(23, 28, 41, 0.8);
            border-right: 1px solid rgba(255,255,255,0.1);
        }

        .metric-card {
            background: rgba(255,255,255,0.05);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            text-align: center;
        }

        .video-box {
            border: 2px solid #262730;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
    <div class="header-box">
        <h1 style="color:white; margin:0; font-size:2.5rem;">
            VISION SAFE AI
        </h1>
        <p style="color:rgba(255,255,255,0.8); margin:5px 0 0 0;">
            Next-Gen Industrial Safety Compliance Monitoring
        </p>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)

    st.title("Control Panel")
    st.markdown("---")

    conf_thresh = st.slider(
        "Detection Sensitivity",
        0.0,
        1.0,
        0.5,
        0.05
    )

    img_size = st.selectbox(
        "Inference Quality",
        [480, 640, 1280],
        index=1
    )

    st.markdown("---")

    st.write("🛰️ **Model Status:** Ready")
    st.write("🛡️ **Protocol:** OSHA 1910")

# --- MAIN LAYOUT ---
col_main, col_info = st.columns([3, 1])

with col_info:
    st.subheader("Live Analytics")

    v_metric = st.empty()

    status_indicator = st.empty()
    status_indicator.info("System: Standby")

    with st.expander("📝 Detected Classes", expanded=True):
        st.caption(
            "Monitoring for: Hardhat, Mask, Vest, "
            "and their respective non-compliance versions."
        )

with col_main:
    st.info("📷 Real-time PPE detection")

    video_placeholder = st.empty()

    start_btn = st.button(
        "🚀 START DETECTION",
        use_container_width=True,
        type="primary"
    )

    stop_btn = st.button(
        "🛑 STOP DETECTION",
        use_container_width=True
    )

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

# --- DETECTION LOGIC ---
if start_btn:

    model = load_model()

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Webcam resolution
    cap.set(3, 1280)
    cap.set(4, 720)

    VIOLATION_CLASSES = [
        'NO-Hardhat',
        'NO-Mask',
        'NO-Safety Vest'
    ]

    status_indicator.warning("System: Webcam Active...")

    while cap.isOpened():

        success, img = cap.read()

        if not success:
            st.error("Failed to access webcam.")
            break

        # YOLO inference
        results = model(
            img,
            imgsz=img_size,
            stream=True
        )

        violations_in_frame = 0

        for r in results:

            for box in r.boxes:

                conf = float(box.conf[0])

                if conf < conf_thresh:
                    continue

                cls = int(box.cls[0])

                currentClass = model.names[cls]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Check violation
                is_violation = currentClass in VIOLATION_CLASSES

                color = (0, 0, 255) if is_violation else (0, 255, 0)

                if is_violation:
                    violations_in_frame += 1

                # Draw bounding box
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Label
                cvzone.putTextRect(
                    img,
                    f'{currentClass} {conf:.2f}',
                    (max(0, x1), max(35, y1)),
                    scale=1.5,
                    thickness=2,
                    colorT=(0, 0, 0),
                    colorR=color,
                    offset=5
                )

        # Display webcam frame
        video_placeholder.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

        # Live metrics
        v_metric.metric(
            "Active Violations",
            violations_in_frame,
            delta="OK" if violations_in_frame == 0
            else f"+ {violations_in_frame} ALERT",
            delta_color="normal"
            if violations_in_frame == 0
            else "inverse"
        )

        # Stop button logic
        if stop_btn:
            break

    cap.release()

    status_indicator.success("System: Detection Stopped")

    st.toast(
        "Webcam detection finished successfully!",
        icon="✅"
    )