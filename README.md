# AI-Powered Surveillance System: Face Recognition and Lip-Reading

This repository contains a college project that demonstrates an AI-powered surveillance system with real-time face recognition and lip-reading capabilities. The project includes a user-friendly Streamlit web application, as well as several command-line scripts for different use cases.

## Features

- **Real-time Face Recognition**: Identify and verify individuals from a live webcam feed or a pre-recorded video file.
- **Lip-Reading**: Predict speech from video by analyzing lip movements (requires a trained model).
- **Suspect Detection**: Highlight and flag individuals who match a provided "suspect" image.
- **Multiple Application Interfaces**: Choose between a comprehensive Streamlit web application, a simpler face-recognition-only app, or command-line scripts.
- **Mobile Camera Integration**: Use your mobile phone's camera as a video source for suspect detection.

## Applications Overview

This repository includes several applications to demonstrate the system's capabilities:

- **`Face_Lip App/combined_ai_surveillance.py`**: The main application. A full-featured Streamlit web app that combines face recognition and lip-reading.
- **`Face_Lip App/face_recognition_streamlit.py`**: A simplified Streamlit app focused solely on face recognition from a webcam, video, or single image.
- **`suspect_detection_with_mobile_camera_feed.py`**: A command-line script that uses a mobile phone's camera feed for suspect detection.
- **`main_face.py`**: A basic command-line script for face recognition using a standard webcam.

## Setup and Usage

### Dependencies

To get started, you need to install the required Python libraries. You can install them using pip and the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Running the Streamlit App

The main application is the combined AI surveillance system. To run it, navigate to the `Face_Lip App` directory and use Streamlit:

```bash
cd "Face_Lip App"
streamlit run combined_ai_surveillance.py
```

This will launch a web application in your browser where you can:
1.  Upload a suspect's image for face recognition.
2.  Optionally, upload a pre-trained lip-reading model (see the training section below).
3.  Choose your input source (webcam or video file).
4.  Start the detection process.

## Lip-Reading Model Training

This repository includes the necessary tools to train your own lip-reading model, as a pre-trained model is not provided.

### Requirements

- A GPU is highly recommended for training the model due to the computational requirements of deep learning.
- The training process is detailed in the `Lip_Reading Real.ipynb` Jupyter notebook.

### Steps to Train

1.  **Open the Notebook**: Launch the `Lip_Reading Real.ipynb` notebook in a Jupyter environment.
2.  **Install Dependencies**: The notebook includes cells to install the necessary libraries.
3.  **Download Data**: The notebook contains a script to download the required dataset from Google Drive. This dataset consists of videos and corresponding alignment files.
4.  **Run the Training Cells**: Execute the cells in the notebook to preprocess the data, build the model, and start the training process.
5.  **Save the Model**: The training script will save model checkpoints (`.h5` files). You can use these saved weights in the `combined_ai_surveillance.py` application.

## Sample Image

The file `WIN_20250804_12_46_54_Pro.jpg` is a sample image of a person that can be used to test the face recognition functionality. You can upload this image as the "suspect" in the Streamlit applications.
