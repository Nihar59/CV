
# Car License Plate Detection

This project implements automatic car license plate detection using YOLO v8, and trained on a custom dataset obtained from Kaggle. The application is deployed using Streamlit for easy user interaction.

## Sample Output
![License Plate Detection](https://github.com/Nihar59/Computer_Vision/assets/69728446/cb91ab84-bdad-4aee-82d1-f05a034f5944)



## Features

- **Upload Media:** Allows users to upload images or videos for license plate detection.
- **Real-time Processing:** Capable of processing both images and videos, displaying results promptly.
- **License Plate Recognition:** Detects license plates in uploaded media.
- **Output Visualization:** Displays annotated images or processed videos with bounding boxes around detected license plates and recognized text overlay.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Nihar59/Computer_Vision
   cd Computer_Vision/License_Plate_Detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Make sure to have the necessary versions of Python and CUDA installed if using GPU. (my output is cpu based)

3. Run the Streamlit app:

   ```bash
   streamlit run yolo_application.py
   ```

   Replace `yolo_application.py` with the name of your Streamlit application script.

## Usage

- Access the Streamlit web interface locally or on a server where the app is deployed.
- Upload an image or video file containing cars with visible license plates.
- Click "Proceed" to start processing.
- The application will display the processed media with annotated bounding box around license plates and confidence percentage.

## Dependencies

- **YOLOv8:** Utilized for object detection and localization.
- **Streamlit:** Framework for deploying and interacting with machine learning applications.
- **OpenCV:** Image and video processing library used for loading, processing, and saving media files.


## Acknowledgments

- **Ultralytics YOLO:** For providing a powerful YOLO implementation.
- **Kaggle:** For the dataset used for training.
- **Streamlit Community:** For the intuitive framework for deploying ML applications.

