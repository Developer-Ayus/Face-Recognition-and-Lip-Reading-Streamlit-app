import cv2
import streamlit as st
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from deepface import DeepFace
import tempfile
import os
from PIL import Image
import time
from typing import List
import matplotlib.pyplot as plt
import imageio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, TimeDistributed

# Set page configuration
st.set_page_config(
    page_title="AI Surveillance System",
    page_icon="🕵🏻",
    layout="wide"
)

# Initialize session state
if 'face_model' not in st.session_state:
    st.session_state.face_model = None
if 'lip_model' not in st.session_state:
    st.session_state.lip_model = None
if 'suspect_image' not in st.session_state:
    st.session_state.suspect_image = None
if 'suspect_path' not in st.session_state:
    st.session_state.suspect_path = None

# Lip reading vocabulary and character mappings
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

@st.cache_resource
def load_face_recognition_model():
    """Load YOLO model for face recognition"""
    try:
        model = YOLO("yolo11n.pt")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

@st.cache_resource
def create_lip_reading_model():
    """Create and return lip reading model architecture"""
    try:
        model = Sequential()
        model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool3D((1,2,2)))

        model.add(Conv3D(256, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool3D((1,2,2)))

        model.add(Conv3D(75, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool3D((1,2,2)))

        model.add(TimeDistributed(Reshape((5 * 17 * 75,))))

        model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
        model.add(Dropout(.5))

        model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
        model.add(Dropout(.5))

        model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))
        
        return model
    except Exception as e:
        st.error(f"Error creating lip reading model: {e}")
        return None

def load_lip_reading_weights(model, weights_path):
    """Load pre-trained weights for lip reading model"""
    try:
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            return True
        else:
            st.warning(f"Weights file not found: {weights_path}")
            return False
    except Exception as e:
        st.error(f"Error loading lip reading weights: {e}")
        return False

def save_uploaded_image(uploaded_file):
    """Save uploaded image to temporary file"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        return temp_file.name
    except Exception as e:
        st.error(f"Error saving uploaded image: {e}")
        return None

def is_suspect(face_crop, suspect_path, threshold=0.4):
    """Check if detected face matches the suspect"""
    try:
        if face_crop.size == 0:
            return False
        
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        result = DeepFace.verify(
            face_rgb, 
            suspect_path, 
            model_name="Facenet", 
            enforce_detection=False,
            distance_metric="cosine"
        )
        
        return result["verified"] and result["distance"] < threshold
    except Exception as e:
        return False

def extract_lip_region(frame, face_bbox):
    """Extract lip region from detected face"""
    try:
        x1, y1, x2, y2 = face_bbox
        face_height = y2 - y1
        face_width = x2 - x1
        
        # Estimate lip region (lower third of face)
        lip_y1 = y1 + int(face_height * 0.6)
        lip_y2 = y1 + int(face_height * 0.9)
        lip_x1 = x1 + int(face_width * 0.2)
        lip_x2 = x1 + int(face_width * 0.8)
        
        # Ensure coordinates are within frame bounds
        lip_y1 = max(0, lip_y1)
        lip_y2 = min(frame.shape[0], lip_y2)
        lip_x1 = max(0, lip_x1)
        lip_x2 = min(frame.shape[1], lip_x2)
        
        lip_region = frame[lip_y1:lip_y2, lip_x1:lip_x2]
        
        # Resize to expected input size for lip reading model
        if lip_region.size > 0:
            lip_region = cv2.resize(lip_region, (140, 46))
            lip_region = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)
            lip_region = np.expand_dims(lip_region, axis=-1)
            return lip_region
        return None
    except Exception as e:
        return None

def process_lip_frames(lip_frames):
    """Process lip frames for lip reading prediction"""
    try:
        if len(lip_frames) == 0:
            return ""
        
        # Pad or truncate to 75 frames
        if len(lip_frames) < 75:
            # Pad with the last frame
            last_frame = lip_frames[-1] if lip_frames else np.zeros((46, 140, 1))
            lip_frames.extend([last_frame] * (75 - len(lip_frames)))
        else:
            lip_frames = lip_frames[:75]
        
        # Normalize frames
        frames_array = np.array(lip_frames)
        mean = np.mean(frames_array)
        std = np.std(frames_array)
        
        if std > 0:
            normalized_frames = (frames_array - mean) / std
        else:
            normalized_frames = frames_array - mean
        
        return normalized_frames.astype(np.float32)
    except Exception as e:
        st.error(f"Error processing lip frames: {e}")
        return None

def predict_lip_reading(model, lip_frames):
    """Predict speech from lip movements"""
    try:
        if model is None or lip_frames is None:
            return ""
        
        # Add batch dimension
        lip_input = np.expand_dims(lip_frames, axis=0)
        
        # Make prediction
        prediction = model.predict(lip_input)
        
        # Decode prediction using CTC
        decoded = tf.keras.backend.ctc_decode(
            prediction, 
            input_length=[75], 
            greedy=True
        )[0][0].numpy()
        
        # Convert to text
        text = tf.strings.reduce_join([num_to_char(word) for word in decoded[0]]).numpy().decode('utf-8')
        return text.strip()
    except Exception as e:
        st.error(f"Error in lip reading prediction: {e}")
        return ""

def process_frame_combined(frame, face_model, lip_model, suspect_path, confidence_threshold=0.5, collect_lip_frames=False, lip_frames_buffer=None):
    """Process frame for both face recognition and lip reading"""
    if face_model is None:
        return frame, False, ""
    
    suspect_detected = False
    lip_text = ""
    
    try:
        # Run YOLO detection
        results = face_model(frame, verbose=False)
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if cls == 0 and conf > confidence_threshold:
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person: {conf:.2f}", (x1, y1 - 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Face recognition
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size > 0 and suspect_path and is_suspect(face_crop, suspect_path):
                            suspect_detected = True
                            cv2.putText(frame, "SUSPECT DETECTED!", (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        
                        # Lip reading
                        if collect_lip_frames and lip_frames_buffer is not None:
                            lip_region = extract_lip_region(frame, (x1, y1, x2, y2))
                            if lip_region is not None:
                                lip_frames_buffer.append(lip_region)
                                
                                # Draw lip region indicator
                                lip_y1 = y1 + int((y2 - y1) * 0.6)
                                lip_y2 = y1 + int((y2 - y1) * 0.9)
                                lip_x1 = x1 + int((x2 - x1) * 0.2)
                                lip_x2 = x1 + int((x2 - x1) * 0.8)
                                cv2.rectangle(frame, (lip_x1, lip_y1), (lip_x2, lip_y2), (255, 0, 0), 2)
                                cv2.putText(frame, "LIP TRACKING", (lip_x1, lip_y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    except Exception as e:
        st.error(f"Error processing frame: {e}")
    
    return frame, suspect_detected, lip_text

def main():
    st.title("🕵🏻 AI Surveillance System")
    st.markdown("**Combined Face Recognition & Lip Reading System**")
    
    # Sidebar configuration
    st.sidebar.header("🔧 System Configuration")
    
    # Model loading section
    st.sidebar.subheader("1. AI Models")
    
    # Load face recognition model
    if st.session_state.face_model is None:
        with st.spinner("Loading Face Recognition Model..."):
            st.session_state.face_model = load_face_recognition_model()
    
    if st.session_state.face_model is not None:
        st.sidebar.success("✅ Face Recognition: Loaded")
    else:
        st.sidebar.error("❌ Face Recognition: Failed")
    
    # Load lip reading model
    st.sidebar.subheader("Lip Reading Model")
    weights_file = st.sidebar.file_uploader(
        "Upload Lip Reading Weights (.h5)", 
        type=['h5'],
        help="Upload pre-trained weights for the lip reading model"
    )
    
    if weights_file is not None and st.session_state.lip_model is None:
        with st.spinner("Loading Lip Reading Model..."):
            # Save weights file temporarily
            temp_weights = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
            temp_weights.write(weights_file.getvalue())
            temp_weights.close()
            
            # Create and load model
            st.session_state.lip_model = create_lip_reading_model()
            if st.session_state.lip_model is not None:
                if load_lip_reading_weights(st.session_state.lip_model, temp_weights.name):
                    st.sidebar.success("✅ Lip Reading: Loaded")
                else:
                    st.session_state.lip_model = None
                    st.sidebar.error("❌ Lip Reading: Failed to load weights")
            
            # Clean up
            os.unlink(temp_weights.name)
    elif st.session_state.lip_model is not None:
        st.sidebar.success("✅ Lip Reading: Loaded")
    else:
        st.sidebar.warning("⚠️ Lip Reading: Upload weights file")
    
    # Suspect image upload
    st.sidebar.subheader("2. Suspect Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose suspect image", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the person to detect"
    )
    
    if uploaded_file is not None:
        suspect_image = Image.open(uploaded_file)
        st.sidebar.image(suspect_image, caption="Suspect Image", width=200)
        
        suspect_path = save_uploaded_image(uploaded_file)
        if suspect_path:
            st.session_state.suspect_path = suspect_path
            st.sidebar.success("✅ Suspect image loaded!")
    
    # Detection settings
    st.sidebar.subheader("3. Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.1
    )
    
    enable_face_recognition = st.sidebar.checkbox("Enable Face Recognition", value=True)
    enable_lip_reading = st.sidebar.checkbox("Enable Lip Reading", value=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎥 Detection Feed")
        
        # Input source selection
        input_source = st.radio(
            "Select Input Source:",
            ["Webcam", "Upload Video"],
            horizontal=True
        )
        
        # Check if models are ready
        if st.session_state.face_model is None:
            st.error("❌ Face Recognition model not loaded")
            return
        
        if enable_face_recognition and st.session_state.suspect_path is None:
            st.warning("⚠️ Please upload a suspect image for face recognition")
        
        if enable_lip_reading and st.session_state.lip_model is None:
            st.warning("⚠️ Please upload lip reading model weights")
        
        # Processing based on input source
        if input_source == "Webcam":
            st.info("Click 'Start Detection' to begin real-time surveillance")
            
            col_start, col_stop = st.columns(2)
            
            with col_start:
                start_detection = st.button("🚀 Start Detection", type="primary")
            
            if start_detection:
                with col_stop:
                    stop_detection = st.button("🛑 Stop Detection", type="secondary")
                
                # Initialize detection
                frame_placeholder = st.empty()
                status_placeholder = st.empty()
                lip_text_placeholder = st.empty()
                
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("❌ Could not open webcam")
                    return
                
                # Lip reading buffer
                lip_frames_buffer = []
                lip_prediction_counter = 0
                
                while not stop_detection:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Failed to read from webcam")
                        break
                    
                    # Process frame
                    processed_frame, suspect_detected, _ = process_frame_combined(
                        frame,
                        st.session_state.face_model,
                        st.session_state.lip_model if enable_lip_reading else None,
                        st.session_state.suspect_path if enable_face_recognition else None,
                        confidence_threshold,
                        enable_lip_reading,
                        lip_frames_buffer
                    )
                    
                    # Convert BGR to RGB for Streamlit
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Status updates
                    if suspect_detected:
                        status_placeholder.error("🚨 SUSPECT DETECTED!")
                    else:
                        status_placeholder.info("🔍 Monitoring...")
                    
                    # Lip reading prediction
                    if enable_lip_reading and st.session_state.lip_model is not None:
                        lip_prediction_counter += 1
                        if lip_prediction_counter >= 75 and len(lip_frames_buffer) >= 75:  # Every 75 frames
                            with st.spinner("Reading lips..."):
                                processed_lip_frames = process_lip_frames(lip_frames_buffer[-75:])
                                if processed_lip_frames is not None:
                                    lip_text = predict_lip_reading(st.session_state.lip_model, processed_lip_frames)
                                    if lip_text:
                                        lip_text_placeholder.success(f"👄 Lip Reading: '{lip_text}'")
                            lip_prediction_counter = 0
                    
                    time.sleep(0.1)
                
                cap.release()
        
        elif input_source == "Upload Video":
            video_file = st.file_uploader("Choose video file", type=['mp4', 'avi', 'mov'])
            
            if video_file is not None:
                # Save video temporarily
                temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_video.write(video_file.read())
                temp_video.close()
                
                # Process video
                cap = cv2.VideoCapture(temp_video.name)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                st.info(f"Processing video with {frame_count} frames...")
                
                frame_placeholder = st.empty()
                progress_bar = st.progress(0)
                results_placeholder = st.empty()
                
                frame_num = 0
                suspect_detections = []
                lip_frames_buffer = []
                all_lip_predictions = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every frame for lip reading, every 10th for display
                    processed_frame, suspect_detected, _ = process_frame_combined(
                        frame,
                        st.session_state.face_model,
                        st.session_state.lip_model if enable_lip_reading else None,
                        st.session_state.suspect_path if enable_face_recognition else None,
                        confidence_threshold,
                        enable_lip_reading,
                        lip_frames_buffer
                    )
                    
                    if suspect_detected:
                        suspect_detections.append(frame_num)
                    
                    # Display every 10th frame
                    if frame_num % 10 == 0:
                        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Lip reading prediction every 75 frames
                    if enable_lip_reading and st.session_state.lip_model is not None and len(lip_frames_buffer) >= 75:
                        if frame_num % 75 == 0:
                            processed_lip_frames = process_lip_frames(lip_frames_buffer[-75:])
                            if processed_lip_frames is not None:
                                lip_text = predict_lip_reading(st.session_state.lip_model, processed_lip_frames)
                                if lip_text.strip():
                                    all_lip_predictions.append(f"Frame {frame_num}: '{lip_text}'")
                    
                    frame_num += 1
                    progress_bar.progress(frame_num / frame_count)
                
                cap.release()
                os.unlink(temp_video.name)
                
                # Display results
                with results_placeholder.container():
                    if suspect_detections:
                        st.success(f"✅ Suspect detected in {len(suspect_detections)} frames!")
                        st.info(f"Detection frames: {suspect_detections[:10]}{'...' if len(suspect_detections) > 10 else ''}")
                    else:
                        st.info("ℹ️ No suspect detected in the video.")
                    
                    if all_lip_predictions:
                        st.subheader(" Lip Reading Results")
                        for prediction in all_lip_predictions[-5:]:  # Show last 5 predictions
                            st.write(prediction)
                        if len(all_lip_predictions) > 5:
                            st.info(f"... and {len(all_lip_predictions) - 5} more predictions")
    
    with col2:
        st.subheader("📊 System Status")
        
        # Model status
        if st.session_state.face_model is not None:
            st.success("✅ Face Recognition: Ready")
        else:
            st.error("❌ Face Recognition: Not Ready")
        
        if st.session_state.lip_model is not None:
            st.success("✅ Lip Reading: Ready")
        else:
            st.warning("⚠️ Lip Reading: Needs Weights")
        
        if st.session_state.suspect_path is not None:
            st.success("✅ Suspect Profile: Loaded")
        else:
            st.warning("⚠️ Suspect Profile: Not Set")
        
        st.subheader("🎯 Active Features")
        if enable_face_recognition:
            st.write("🔍 Face Recognition: ON")
        if enable_lip_reading:
            st.write("👄 Lip Reading: ON")
        
        st.subheader("📋 Instructions")
        st.markdown("""
        **Setup:**
        1. Upload suspect image (for face recognition)
        2. Upload lip reading weights (optional)
        3. Select input source
        4. Enable desired features
        
        **Tips:**
        - Use clear, front-facing images
        - Ensure good lighting
        - Lip reading works best with clear mouth visibility
        - Higher confidence = fewer false positives
        """)
        
        st.subheader("⚙️ Technical Info")
        st.markdown("""
        **Models:**
        - YOLOv11: Person detection
        - DeepFace: Face recognition
        - CNN+LSTM: Lip reading
        
        **Features:**
        - Real-time processing
        - Multi-modal AI analysis
        - Customizable thresholds
        """)

if __name__ == "__main__":
    main()
