# Import necessary libraries
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Define different arrays to handle color points for each color
teal_points = [deque(maxlen=1024)]
yellow_points = [deque(maxlen=1024)]
purple_points = [deque(maxlen=1024)]
green_points = [deque(maxlen=1024)]

# Initialize indexes for each color
teal_index = 0
yellow_index = 0
purple_index = 0
green_index = 0

# Define the kernel for dilation
dilation_kernel = np.ones((5, 5), np.uint8)

# Define custom colors with names and RGB values
custom_colors = [
    {"name": "TEAL", "rgb": (128, 128, 0)},
    {"name": "YELLOW", "rgb": (0, 255, 255)},
    {"name": "PURPLE", "rgb": (128, 0, 128)},
    {"name": "GREEN", "rgb": (0, 255, 0)}
]

# Extract the RGB values from the color list
colors = [color["rgb"] for color in custom_colors]

# Initialize color index
selected_color_index = 0

# Create the painting canvas
painting_canvas = np.zeros((471, 636, 3)) + 255
painting_canvas = cv2.rectangle(painting_canvas, (40, 1), (140, 65), (0, 0, 0), 2)
painting_canvas = cv2.rectangle(painting_canvas, (160, 1), (255, 65), (128, 128, 0), 2)
painting_canvas = cv2.rectangle(painting_canvas, (275, 1), (370, 65), (0, 255, 255), 2)
painting_canvas = cv2.rectangle(painting_canvas, (390, 1), (485, 65), (128, 0, 128), 2)
painting_canvas = cv2.rectangle(painting_canvas, (505, 1), (600, 65), (0, 255, 0), 2)

cv2.putText(painting_canvas, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(painting_canvas, "TEAL", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(painting_canvas, "YELLOW", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(painting_canvas, "PURPLE", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(painting_canvas, "GREEN", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True
while ret:
    # Read a frame from the webcam
    ret, frame = cap.read()

    frame_height, frame_width, _ = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (128, 128, 0), 2)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 255), 2)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (128, 0, 128), 2)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 0), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "TEAL", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "PURPLE", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Get hand landmark predictions
    result = hands.process(frame_rgb)

    # Post-process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmark_x = int(landmark.x * frame_width)
                landmark_y = int(landmark.y * frame_height)
                landmarks.append([landmark_x, landmark_y])

            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, 3, (0, 255, 0), -1)

        if thumb[1] - center[1] < 30:
            teal_points.append(deque(maxlen=512))
            teal_index += 1
            yellow_points.append(deque(maxlen=512))
            yellow_index += 1
            purple_points.append(deque(maxlen=512))
            purple_index += 1
            green_points.append(deque(maxlen=512))
            green_index += 1

        elif center[1] <= 65:
            if 40 <= center[0] <= 140:  # Clear Button
                teal_points = [deque(maxlen=512)]
                yellow_points = [deque(maxlen=512)]
                purple_points = [deque(maxlen=512)]
                green_points = [deque(maxlen=512)]

                teal_index = 0
                yellow_index = 0
                purple_index = 0
                green_index = 0

                painting_canvas[67:, :, :] = 255
            elif 160 <= center[0] <= 255:
                selected_color_index = 1  # Teal
            elif 275 <= center[0] <= 370:
                selected_color_index = 2  # Yellow
            elif 390 <= center[0] <= 485:
                selected_color_index = 3  # Purple
            elif 505 <= center[0] <= 600:
                selected_color_index = 4  # Green
        else:
            if selected_color_index == 1:
                teal_points[teal_index].appendleft(center)
            elif selected_color_index == 2:
                yellow_points[yellow_index].appendleft(center)
            elif selected_color_index == 3:
                purple_points[purple_index].appendleft(center)
            elif selected_color_index == 4:
                green_points[green_index].appendleft(center)
    
    # Append the next deques when nothing is detected to avoid messing up
    else:
        teal_points.append(deque(maxlen=512))
        teal_index += 1
        yellow_points.append(deque(maxlen=512))
        yellow_index += 1
        purple_points.append(deque(maxlen=512))
        purple_index += 1
        green_points.append(deque(maxlen=512))
        green_index += 1

    # Draw lines of all the colors on the canvas and frame
    color_point_lists = [teal_points, yellow_points, purple_points, green_points]
    for i, color_points in enumerate(color_point_lists):
        for j in range(len(color_points)):
            for k in range(1, len(color_points[j])):
                if color_points[j][k - 1] is None or color_points[j][k] is None:
                    continue
                cv2.line(frame, color_points[j][k - 1], color_points[j][k], colors[i], 2)
                cv2.line(painting_canvas, color_points[j][k - 1], color_points[j][k], colors[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", painting_canvas)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()