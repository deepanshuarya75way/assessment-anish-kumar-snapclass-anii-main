import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import mediapipe as mp
from src.database.db import enroll_student_to_subject
from src.database.config import supabase
from PIL import Image
import time

@st.dialog("check liveliness of a student")
mp_face_mesh = mp.solutions.face_mesh

# Indices for Eye Landmarks (MediaPipe Face Mesh architecture)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    """Calculate Eye Aspect Ratio (EAR) to detect blinks."""
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))

         # Verticals: p2-p6, p3-p5
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    # Horizontal: p1-p4
    h = np.linalg.norm(coords[0] - coords[3])
    
    ear = (v1 + v2) / (2.0 * h)
    return ear

class LivenessTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True, 
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.6
        )
        # Blink detection variables
        self.ear_threshold = 0.22  # Adjust based on your lighting/camera distance
        self.blink_counter = 0
        self.is_blinked = False
        self.liveness_confirmed = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr2x")
        h, w, _ = img.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)
        
        status_text = "Scan Face"
        color = (0, 165, 255) # Orange

        if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate EAR for both eyes
        left_ear = calculate_ear(face_landmarks, LEFT_EYE, w, h)
        right_ear = calculate_ear(face_landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Blink logic tracking
        if avg_ear < self.ear_threshold:
            if not self.is_blinked:
                self.is_blinked = True
        else:
            if self.is_blinked:
                self.blink_counter += 1
                self.is_blinked = False

         # Mark liveness confirmed after 2 genuine blinks
            if self.blink_counter >= 2:
                self.liveness_confirmed = True
                status_text = "Liveness Verified!"
                color = (0, 255, 0) # Green
            else:
                status_text = f"Blink Counter: {self.blink_counter}/2"
                color = (0, 215, 255) # Yellow

            # Visual overlay on camera stream
            cv2.putText(img, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(img, f"EAR: {avg_ear:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(img, "No Face Detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return img

@st.dialog("Capture or upload photos")
def add_photos_dialog():

    st.write('Add classroom photos to scan for attendance')

    if 'photo_tab' not in st.session_state:
        st.session_state.photo_tab = 'camera'

    t1, t2 = st.columns(2)

    with t1:
        type_camera = "primary" if st.session_state.photo_tab == 'camera' else 'tertiary'
        if st.button('Camera', type=type_camera, width='stretch'):
            st.session_state.photo_tab = 'camera'
    
    with t2:
        type_upload = "primary" if st.session_state.photo_tab == 'upload' else 'tertiary'
        if st.button('Upload photos', type=type_upload, width='stretch'):
            st.session_state.photo_tab = 'upload'

    
    if st.session_state.photo_tab == 'camera':
        cam_photo = st.camera_input('Take Snapshot', key='dialog_cam')
        if cam_photo:
            st.session_state.attendance_images.append(Image.open(cam_photo))
            st.toast('Photo Captured')
            st.rerun()

    if st.session_state.photo_tab == 'upload':
        uploaded_files = st.file_uploader('choose image files', type=['jpg', 'png', 'jpeg'], accept_multiple_files=True, key='dialog_upload')

        if uploaded_files:
            for f in uploaded_files:
                st.session_state.attendance_images.append(Image.open(f))
            
            st.toast('Photo Uploaded Successfully')
            st.rerun()

    st.divider()
    if st.button('Done', type='primary', width='stretch'):
        st.rerun()