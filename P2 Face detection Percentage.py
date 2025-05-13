import cv2
import time
import mediapipe as mp

# Initialize the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Variables to calculate FPS
ptime = 0  # previous time

# Initialize MediaPipe face detection and drawing utilities
mpFace = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFace.FaceDetection(min_detection_confidence=0.6)  # confidence threshold

while True:
    # Read a frame from the webcam
    success, img = cap.read()
    if not success:
        break  # Exit if frame not captured properly

    # Convert BGR (OpenCV default) to RGB (MediaPipe expects RGB)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = faceDetection.process(imgRGB)

    # If any face detections are found
    if results.detections:
        for id, det in enumerate(results.detections):
            # Get the relative bounding box coordinates
            bbox = det.location_data.relative_bounding_box

            # Get image dimensions to convert relative bbox to pixel values
            h, w, c = img.shape
            bb = int(bbox.xmin * w), int(bbox.ymin * h), \
                 int(bbox.width * w), int(bbox.height * h)

            # Draw a rectangle around the detected face
            cv2.rectangle(img, bb, (255, 219, 50), 2)

            # Draw the confidence score above the rectangle
            cv2.putText(img, f'{int(det.score[0] * 100)}%',
                        (bb[0], bb[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        2, (230, 199, 0), 3)

    # Calculate and display the Frames Per Second (FPS)
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    # Draw the FPS on the top-left corner of the image
    cv2.putText(img, f'FPS: {int(fps)}',
                (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (36, 23, 21), 3)

    # Show the result in a window titled "Image"
    cv2.imshow('Image', img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
