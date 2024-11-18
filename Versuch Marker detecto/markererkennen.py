import cv2
import numpy as np

# Video path
video_path = "marker weiss.mp4"

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

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Allow user to select the marker
roi = cv2.selectROI("Select Marker", prev_frame, fromCenter=False, showCrosshair=True)
x, y, w, h = map(int, roi)
prev_points = np.array([[x + w // 2, y + h // 2]], dtype=np.float32).reshape(-1, 1, 2)

cv2.destroyWindow("Select Marker")

# Initialize variables for tracking highest and lowest points
max_y = -float('inf')
min_y = float('inf')
max_frame = None
min_frame = None

# Resize parameters for the display window
resize_width = 640  # Set the new width of the window
resize_height = 480  # Set the new height of the window

# Process frames with optical flow
frame_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)

    # Update tracking position if successful
    if status[0][0] == 1:
        x, y = next_points[0][0]
        cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), -1)

        # Update previous frame and points
        prev_gray = gray.copy()
        prev_points = next_points

        # Track the Y-coordinate (vertical position) of the marker
        if y > max_y:
            max_y = y
            max_frame = frame.copy()  # Save the frame where marker is at the highest point
            max_frame_counter = frame_counter

        if y < min_y:
            min_y = y
            min_frame = frame.copy()  # Save the frame where marker is at the lowest point
            min_frame_counter = frame_counter

    # Resize the frame to fit the window while maintaining the aspect ratio
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    new_width = resize_width
    new_height = int(new_width / aspect_ratio)

    if new_height > resize_height:
        new_height = resize_height
        new_width = int(new_height * aspect_ratio)

    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Display the resized frame
    cv2.imshow("Tracking", resized_frame)

    frame_counter += 1

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save the frames with the highest and lowest marker positions
if max_frame is not None:
    cv2.imwrite("highest_point_frame.jpg", max_frame)
    print(f"Saved frame at highest point (frame {max_frame_counter})")

if min_frame is not None:
    cv2.imwrite("lowest_point_frame.jpg", min_frame)
    print(f"Saved frame at lowest point (frame {min_frame_counter})")

# Display both frames side by side without stretching them
if max_frame is not None and min_frame is not None:
    # Get the height and width of the frames
    max_height, max_width = max_frame.shape[:2]
    min_height, min_width = min_frame.shape[:2]

    # Calculate the height for both frames to be the same
    new_height = max(max_height, min_height)
    max_resized = max_frame.copy()
    min_resized = min_frame.copy()

    # Resize the frames to have the same height (preserving aspect ratio)
    max_resized = cv2.resize(max_resized, (int(max_width * new_height / max_height), new_height))
    min_resized = cv2.resize(min_resized, (int(min_width * new_height / min_height), new_height))

    # Pad the frames to match the height (in case the width is different)
    if max_resized.shape[1] < min_resized.shape[1]:
        pad_width = min_resized.shape[1] - max_resized.shape[1]
        max_resized = cv2.copyMakeBorder(max_resized, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    elif min_resized.shape[1] < max_resized.shape[1]:
        pad_width = max_resized.shape[1] - min_resized.shape[1]
        min_resized = cv2.copyMakeBorder(min_resized, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Stack the frames horizontally without stretching
    combined_frame = np.hstack((max_resized, min_resized))

    # Display the combined frame
    cv2.imshow("Highest and Lowest Point Frames", combined_frame)
    cv2.waitKey(0)  # Wait until any key is pressed to close the window

cv2.destroyAllWindows()
