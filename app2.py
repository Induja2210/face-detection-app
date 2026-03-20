import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from transformers import pipeline

# --------------------- Page Configuration ---------------------
st.set_page_config(
    page_title="Face Detection, Classification & Recognition",
    layout="wide"
)

# --------------------- CSS Styling ---------------------
st.markdown("""
<style>
h1 {text-align: center; font-size: 2.5rem; font-weight: 600;}
p {text-align: center; font-size: 1.05rem;}
.sidebar .sidebar-content {font-size: 0.9rem;}
.result-box {padding: 1rem; border-radius: 10px; background:#eef2f7; margin-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# --------------------- Session State ---------------------
if "known_encodings" not in st.session_state:
    st.session_state.known_encodings = []   # (name, embedding)

# --------------------- Load Models ---------------------
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
                  thresholds=[0.6, 0.7, 0.7], factor=0.709,
                  post_process=True, device=device)

    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    age_pipe = pipeline("image-classification", model="nateraw/vit-age-classifier",
                        device=0 if torch.cuda.is_available() else -1)
    gender_pipe = pipeline("image-classification", model="dima806/fairface_gender_image_detection",
                           device=0 if torch.cuda.is_available() else -1)
    emotion_pipe = pipeline("image-classification", model="mo-thecreator/vit-Facial-Expression-Recognition",
                            device=0 if torch.cuda.is_available() else -1)

    return mtcnn, resnet, age_pipe, gender_pipe, emotion_pipe, device

mtcnn, resnet, age_pipe, gender_pipe, emotion_pipe, device = load_models()

# --------------------- Title ---------------------
st.markdown("<h1>Face Detection • Classification • Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload or capture an image to predict Age, Gender, Emotion and identify known individuals.</p>", unsafe_allow_html=True)

# --------------------- Sidebar ---------------------
with st.sidebar:
    st.header("Settings")
    enable_age = st.checkbox("Predict Age", True)
    enable_gender = st.checkbox("Predict Gender", True)
    enable_emotion = st.checkbox("Predict Emotion", True)
    enable_recognition = st.checkbox("Enable Face Recognition", True)

    st.divider()
    st.subheader("Known Faces Database")
    if st.session_state.known_encodings:
        for nm, _ in st.session_state.known_encodings:
            st.write("•", nm)
    else:
        st.write("No faces registered.")

    with st.expander("Add a Person to Database"):
        new_name = st.text_input("Person's Name")
        new_img = st.file_uploader("Upload Face Image", type=["jpg", "png", "jpeg"])
        if st.button("Register"):
            if new_name and new_img:
                img = Image.open(new_img).convert("RGB")
                aligned = mtcnn(img)
                if aligned is not None:
                    emb = resnet(aligned.unsqueeze(0).to(device)).detach().cpu().numpy().flatten()
                    st.session_state.known_encodings.append((new_name, emb))
                    st.success(f"{new_name} added successfully.")
                else:
                    st.error("Face not detected in the image.")
            else:
                st.error("Name and image are required.")

    if st.button("Clear Database"):
        st.session_state.known_encodings.clear()
        st.success("Database cleared.")

# --------------------- Input Controls ---------------------
st.divider()
input_method = st.radio("Select Input Source", ["Upload Image", "Camera Capture"], horizontal=True)

uploaded_img = None
if input_method == "Upload Image":
    uploaded_img = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
else:
    cam_img = st.camera_input("Capture Image")
    uploaded_img = cam_img if cam_img else None

# --------------------- Processing ---------------------
if uploaded_img:
    img = Image.open(uploaded_img).convert("RGB")
    np_img = np.array(img)
    display_frame = np_img.copy()

    boxes, _ = mtcnn.detect(img)

    if boxes is None:
        st.warning("No faces detected.")
        st.stop()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face = img.crop((x1, y1, x2, y2))
        aligned = mtcnn(face)
        label = ""

        # Face Recognition
        if enable_recognition and aligned is not None and st.session_state.known_encodings:
            emb = resnet(aligned.unsqueeze(0).to(device)).detach().cpu().numpy().flatten()
            distances = [np.linalg.norm(emb - ref) for _, ref in st.session_state.known_encodings]
            idx = int(np.argmin(distances))
            if distances[idx] < 1.0:
                label += f" {st.session_state.known_encodings[idx][0]} |"

        # Age
        if enable_age:
            pred = age_pipe(face)[0]["label"]
            label += f" Age: {pred} |"

        # Gender
        if enable_gender:
            pred = gender_pipe(face)[0]["label"]
            label += f" {pred} |"

        # Emotion
        if enable_emotion:
            pred = emotion_pipe(face)[0]["label"]
            label += f" {pred}"

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(display_frame, label[:60], (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 180, 0), 2)

    st.image(display_frame, caption="Processed Output", use_column_width=True)
    st.success("Processing complete.")
