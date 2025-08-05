import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import tempfile
import os
from PIL import Image
import time

# Set page configuration
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'suspect_image' not in st.session_state:
    st.session_state.suspect_image = None
if 'suspect_path' not in st.session_state:
    st.session_state.suspect_path = None

@st.cache_resource
def load_yolo_model():
    """Load YOLO model with caching"""
    try:
        model = YOLO("yolo11n.pt")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

def save_uploaded_image(uploaded_file):
    """Save uploaded image to temporary file"""
    try:
        # Create temporary file
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
        
        # Convert BGR to RGB for DeepFace
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Compare face with suspect image
        result = DeepFace.verify(
            face_rgb, 
            suspect_path, 
            model_name="Facenet", 
            enforce_detection=False,
            distance_metric="cosine"
        )
        
        # Use both verification result and distance threshold
        return result["verified"] and result["distance"] < threshold
    except Exception as e:
        return False

def process_frame(frame, model, suspect_path, confidence_threshold=0.5):
    """Process a single frame for face detection and recognition"""
    if model is None or suspect_path is None:
        return frame, False
    
    suspect_detected = False
    
    try:
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Only process detected people with sufficient confidence
                    if cls == 0 and conf > confidence_threshold:
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add confidence score
                        cv2.putText(frame, f"Person: {conf:.2f}", (x1, y1 - 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Extract face region
                        face_crop = frame[y1:y2, x1:x2]
                        
                        # Check if this person is the suspect
                        if face_crop.size > 0 and is_suspect(face_crop, suspect_path):
                            suspect_detected = True
                            cv2.putText(frame, "SUSPECT DETECTED!", (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    except Exception as e:
        st.error(f"Error processing frame: {e}")
    
    return frame, suspect_detected

def main():
    st.title("🔍 Face Recognition System")
    st.markdown("Upload a suspect image and use webcam or video file for real-time detection")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Upload suspect image
    st.sidebar.subheader("1. Upload Suspect Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose suspect image", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the person to detect"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        suspect_image = Image.open(uploaded_file)
        st.sidebar.image(suspect_image, caption="Suspect Image", width=200)
        
        # Save to temporary file
        suspect_path = save_uploaded_image(uploaded_file)
        if suspect_path:
            st.session_state.suspect_path = suspect_path
            st.sidebar.success("✅ Suspect image loaded successfully!")
    
    # Model settings
    st.sidebar.subheader("2. Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Person Detection Confidence", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.1,
        help="Minimum confidence for person detection"
    )
    
    # Load YOLO model
    if st.session_state.model is None:
        with st.spinner("Loading YOLO model..."):
            st.session_state.model = load_yolo_model()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Detection Feed")
        
        # Input source selection
        input_source = st.radio(
            "Select Input Source:",
            ["Webcam", "Upload Video", "Upload Image"],
            horizontal=True
        )
        
        if st.session_state.model is None:
            st.error("❌ YOLO model failed to load. Please check your installation.")
            return
        
        if st.session_state.suspect_path is None:
            st.warning("⚠️ Please upload a suspect image first.")
            return
        
        # Process based on input source
        if input_source == "Webcam":
            st.info("Click 'Start Webcam' to begin real-time detection")
            
            if st.button("Start Webcam", type="primary"):
                # Webcam processing
                frame_placeholder = st.empty()
                stop_button = st.button("Stop")
                
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("❌ Could not open webcam")
                    return
                
                while not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Failed to read from webcam")
                        break
                    
                    # Process frame
                    processed_frame, suspect_detected = process_frame(
                        frame, 
                        st.session_state.model, 
                        st.session_state.suspect_path,
                        confidence_threshold
                    )
                    
                    # Convert BGR to RGB for Streamlit
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    frame_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Alert if suspect detected
                    if suspect_detected:
                        st.error("🚨 SUSPECT DETECTED!")
                    
                    time.sleep(0.1)  # Small delay to prevent overwhelming
                
                cap.release()
        
        elif input_source == "Upload Video":
            video_file = st.file_uploader("Choose video file", type=['mp4', 'avi', 'mov'])
            
            if video_file is not None:
                # Save video to temporary file
                temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_video.write(video_file.read())
                temp_video.close()
                
                # Process video
                cap = cv2.VideoCapture(temp_video.name)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                st.info(f"Processing video with {frame_count} frames...")
                
                frame_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                frame_num = 0
                suspect_detections = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every 10th frame for performance
                    if frame_num % 10 == 0:
                        processed_frame, suspect_detected = process_frame(
                            frame,
                            st.session_state.model,
                            st.session_state.suspect_path,
                            confidence_threshold
                        )
                        
                        if suspect_detected:
                            suspect_detections.append(frame_num)
                        
                        # Display frame
                        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                    
                    frame_num += 1
                    progress_bar.progress(frame_num / frame_count)
                
                cap.release()
                os.unlink(temp_video.name)  # Clean up temp file
                
                # Show results
                if suspect_detections:
                    st.success(f"✅ Suspect detected in {len(suspect_detections)} frames!")
                    st.info(f"Detection frames: {suspect_detections}")
                else:
                    st.info("ℹ️ No suspect detected in the video.")
        
        elif input_source == "Upload Image":
            image_file = st.file_uploader("Choose image file", type=['jpg', 'jpeg', 'png'])
            
            if image_file is not None:
                # Load and process image
                image = Image.open(image_file)
                frame = np.array(image)
                
                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Process image
                processed_frame, suspect_detected = process_frame(
                    frame,
                    st.session_state.model,
                    st.session_state.suspect_path,
                    confidence_threshold
                )
                
                # Convert back to RGB for display
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display result
                st.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                
                if suspect_detected:
                    st.error("🚨 SUSPECT DETECTED IN IMAGE!")
                else:
                    st.info("ℹ️ No suspect detected in the image.")
    
    with col2:
        st.subheader("System Status")
        
        # Model status
        if st.session_state.model is not None:
            st.success("✅ YOLO Model: Loaded")
        else:
            st.error("❌ YOLO Model: Not loaded")
        
        # Suspect image status
        if st.session_state.suspect_path is not None:
            st.success("✅ Suspect Image: Loaded")
        else:
            st.warning("⚠️ Suspect Image: Not loaded")
        
        st.subheader("Instructions")
        st.markdown("""
        1. **Upload** a clear image of the suspect
        2. **Choose** your input source (webcam, video, or image)
        3. **Adjust** detection confidence if needed
        4. **Start** the detection process
        
        **Tips:**
        - Use clear, front-facing suspect images
        - Ensure good lighting conditions
        - Higher confidence = fewer false positives
        """)
        
        st.subheader("System Requirements")
        st.markdown("""
        - **OpenCV**: Image processing
        - **YOLO**: Person detection
        - **DeepFace**: Face recognition
        - **Webcam**: For live detection
        """)

if __name__ == "__main__":
    main()
