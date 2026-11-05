import cv2
import numpy as np
import tempfile
import subprocess
import streamlit as st
import imageio_ffmpeg

st.set_page_config(page_title="Road Lane Detection", layout="wide")
st.title("🚗 Road Lane Detection (OpenCV)")

st.markdown("Upload a road video and the app will detect lanes in real-time.")

# File uploader
uploaded_file = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov"])

# Function to process video and detect lanes
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[(int(width * 0.05), height), (int(width * 0.95), height),
                             (int(width * 0.55), int(height * 0.6)),
                             (int(width * 0.45), int(height * 0.6))]], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked, 2, np.pi/180, 100,
                                np.array([]), minLineLength=40, maxLineGap=50)

        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):   # <-- error fix
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        out.write(frame)

    cap.release()
    out.release()

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()                                          # <-- close so OpenCV can read it

    raw_output = "output_raw.mp4"
    output_path = "output.mp4"
    process_video(tfile.name, raw_output)

    # Re-encode to H.264 so the browser can play it
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", raw_output,
                    "-vcodec", "libx264", "-pix_fmt", "yuv420p", output_path],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    st.success("✅ Processing complete! Here’s the output:")
    st.video(output_path)