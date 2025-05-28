import cv2
from ultralytics import YOLO
from gtts import gTTS
import pygame
import os
import time

# Initialize pygame for audio playback
pygame.mixer.init()

# Function to convert text to speech
def text_to_speech(text):
    """
    Converts the given text to speech and plays it.
    """
    print(text)  # Print the text to the console
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save("output.mp3")

    try:
        # Load and play the audio file using pygame
        pygame.mixer.music.load("output.mp3")
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    finally:
        # Ensure the file is deleted even if an error occurs
        if os.path.exists("output.mp3"):
            try:
                pygame.mixer.music.stop()  # Stop the music playback
                pygame.mixer.music.unload()  # Unload the music file
                os.remove("output.mp3")  # Delete the file
            except PermissionError:
                # If the file is still in use, wait and try again
                time.sleep(1)
                if os.path.exists("output.mp3"):
                    os.remove("output.mp3")

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy

# Open the webcam
cap = cv2.VideoCapture(0)  # Use "0" for the default webcam

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera. Please check the camera index and permissions.")
    exit()

# Track already detected objects
detected_objects = set()

# List to store all detected items (unique)
all_detected_items = []

# Main loop
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from the camera.")
        break

    # Perform object detection
    results = model.predict(frame, conf=0.5, verbose=False)  # Disable verbose logs

    # Reset detected objects for the current frame
    current_frame_objects = set()

    # Draw bounding boxes and labels on the frame
    for result in results:
        for box in result.boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get the class label and confidence score
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Add the detected object to the current frame set
            current_frame_objects.add(class_name)

    # Check for new objects in the current frame
    new_objects = current_frame_objects - detected_objects
    if new_objects:
        for obj in new_objects:
            detection_text = f"I see a {obj}"
            print(detection_text)  # Print to console
            text_to_speech(detection_text)  # Speak the text

            # Add the new object to the list of all detected items
            all_detected_items.append(obj)

    # Update the set of detected objects
    detected_objects.update(current_frame_objects)

    # Display the frame with detected objects
    cv2.imshow("Object Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

# Print all detected items at the end
print("\nAll detected items:")
print(all_detected_items)