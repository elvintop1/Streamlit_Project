import cv2
import streamlit as st
import numpy as np
from typing import Any
from PIL import Image

MODEL: str = "./model/MobileNetSSD_deploy.caffemodel"
PROTOTXT: str = "./model/MobileNetSSD_deploy.prototxt.txt"

def process_image(image: np.ndarray) -> np.ndarray:
    """Process an image for object using a pre-trained model

    Args:
        image (np.ndarray): The input image should be in BGR format


    Returns:
        np.ndarray: The detections from the object detection model
    """
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    blob: np.ndarray = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),  # the image
        0.007843,  # scale factor the images channels, scales it down by a factor of 1/n
        (300, 300),  # target size
        127.5,  # mean subtracting the value
    )

    print("Blob: ", blob, "\n")
    net: Any = cv2.dnn.readNetFromCaffe(prototxt=PROTOTXT, caffeModel=MODEL)
    print("\n Net: ", net, "\n")
    net.setInput(blob)
    detections: np.ndarray = net.forward()
    print("Detections: ", detections, "\n")
    return detections

def annotate_image(image: np.ndarray, detections: np.ndarray, confidence_threshold: float = 0.5) -> np.ndarray:
    """
    Annotate the image with bounding boxes for detected objects.

    Args:
        image (np.ndarray): The input image in BGR format.
        detections (np.ndarray): The detections from the object detection model.
        confidence_threshold (float, optional): The threshold for filtering weak detections. Defaults to 0.5.

    Returns:
        np.ndarray: The annotated image.
    """

    (height, width) = image.shape[:2]
    print("Image.shape", image.shape, "\n")
    print("Image: ", image, "\n")
    for i in np.arange(0, detections.shape[2]):
        confidence: float = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            
            box: np.ndarray = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
    return image

def main():
    st.title("Object Detection for Images")
    file: Any = st.file_uploader("Upload image", type=['jpg', 'png', 'jpeg'])
    if file is not None:
        st.image(file, caption="Uploaded Image")

        image: np.ndarray = np.array(Image.open(file))
        detections: np.ndarray = process_image(image)
        processed_image: np.ndarray = annotate_image(image, detections=detections)
        st.image(processed_image, caption="Processed Image")

if __name__ == "__main__":
    main()
