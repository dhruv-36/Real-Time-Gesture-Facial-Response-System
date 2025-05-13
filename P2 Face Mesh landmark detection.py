import cv2
import mediapipe as mp
import time

# Initialize time for FPS calculation
ptime = 0

# Open default webcam
cap = cv2.VideoCapture(0)

# Load MediaPipe utilities
mpDraw = mp.solutions.drawing_utils                 # For drawing facial landmarks
mpMesh = mp.solutions.face_mesh                     # Face mesh model

# Initialize the FaceMesh model
facemesh = mpMesh.FaceMesh(max_num_faces=1)         # Detect only one face

# Define drawing specifications for landmarks and connections
drawSpecs = mpDraw.DrawingSpec(thickness=2, circle_radius=1)

# def __init__(self,
#              static_image_mode=False,
#              max_num_faces=1, #increase if multiple faces to detect
#              refine_landmarks=False,
#              min_detection_confidence=0.5,
#              min_tracking_confidence=0.5):

# Main loop to read and process video frames
while True:
    suc, img = cap.read()                           # Read frame from webcam
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Convert image to RGB for MediaPipe

    results = facemesh.process(imgRGB)              # Run face mesh detection

    if results.multi_face_landmarks:                # If at least one face is detected
        for faceLms in results.multi_face_landmarks:
            # Draw the facial landmarks on the original image
            mpDraw.draw_landmarks(img, faceLms, mpMesh.FACEMESH_CONTOURS,
                                  drawSpecs, drawSpecs)

    # Calculate and display FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    # Draw FPS on the frame
    cv2.putText(img, str(int(fps)), (10, 60),
                cv2.FONT_HERSHEY_PLAIN, 3, (102, 123, 155), 3)

    # Show the frame in a window titled 'Image'
    cv2.imshow('Image', img)

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
