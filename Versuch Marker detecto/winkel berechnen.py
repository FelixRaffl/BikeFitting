import cv2
import numpy as np
import math

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    # Vector AB and BC
    ab = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    # Calculate the angle using the dot product formula
    cos_theta = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cos_theta))

    return angle

# Mouse callback function to capture the points
points = []
def click_event(event, x, y, flags, param):
    global points, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Mark the point
        cv2.imshow("Frame", frame)

        # If 3 points are selected, calculate the angle and draw the lines
        if len(points) == 3:
            # Sort the points to make sure we have ankle, knee, and hip
            ankle, knee, hip = sorted(points, key=lambda p: p[1])

            # Calculate the angle at the knee
            angle = calculate_angle(ankle, knee, hip)
            
            # Draw lines for ankle, knee, and hip
            cv2.line(frame, ankle, knee, (0, 255, 0), 2)  # Ankle to Knee
            cv2.line(frame, knee, hip, (0, 255, 0), 2)    # Knee to Hip
            cv2.putText(frame, f"Angle: {angle:.2f} degrees", (knee[0] + 10, knee[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Frame", frame)

# Load the two extracted frames
frame1 = cv2.imread(r"Versuch Marker detecto\highest_point_frame.jpg")
frame2 = cv2.imread(r"Versuch Marker detecto\lowest_point_frame.jpg")

# Function to let the user select and work with frames
def work_with_frame(selected_frame):
    global points, frame
    points = []  # Reset points for each new frame
    frame = selected_frame.copy()  # Copy the selected frame for processing

    # Display the frame and allow the user to click on the points
    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", click_event)

    # Wait for user to click 3 points
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Work with the first frame
work_with_frame(frame1)

# Work with the second frame
work_with_frame(frame2)
