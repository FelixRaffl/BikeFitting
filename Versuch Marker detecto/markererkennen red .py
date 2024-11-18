import cv2
import numpy as np

# Video path
video_path = "marker rot.mp4"

# Load video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read video.")
    cap.release()
    exit()

# Convert the first frame to HSV
prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)

# Define the red color range in HSV
# Lower and upper bounds for red color in HSV
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Create masks for red color
mask1 = cv2.inRange(prev_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(prev_hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# Find contours of the red regions
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# If any contours are found, get the center of the largest contour (red marker)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        prev_points = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
    else:
        print("Error: Could not find the red marker center.")
        cap.release()
        exit()

cv2.destroyAllWindows()

# Process frames with optical flow
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for red color in the current frame
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours of the red regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If any contours are found, track the red marker's center
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Draw a circle around the red marker
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            # Update the previous points for optical flow tracking
            prev_points = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)

    # Display the frame
    cv2.imshow("Tracking Red Marker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
