import cv2  # import opencv
import mediapipe as mp  # give mediapipe solutions
import numpy as np  # help with trig

mp_drawing = mp.solutions.drawing_utils  # drawing utilities, visualizing poses
mp_pose = mp.solutions.pose  # importing pose estimation models, grabbing pose estimation model


# Create function to calculate angles
def calculate_angle(a, b, c):
    a = np.array(a)  # first
    b = np.array(b)  # mid
    c = np.array(c)  # endpoint

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])  # change to radians
    angle = np.abs(radians * 180 / np.pi)  # angle, to get absolute value

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Video Feed from webcam
# MAKE DETECTIONS
cap = cv2.VideoCapture(0)

counter = 0
stage = None


# set up mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        # Recolour image, when we pass image, we want image to be RGB, automatically image is BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # grab frame and recolour
        image.flags.writeable = False  # save memory

        results = pose.process(image)  # making the detection, process we get detections

        image.flags.writeable = True  # writeable status to true
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # recolour back to BGR

        # EXTRACT LANDMARKS
        try:
            landmarks = results.pose_landmarks.landmark

            # get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # visualize
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Curl counter logic
            if angle > 160:
                stage = 'down'
            if angle < 30 and stage == 'down':
                stage = 'up'
                counter += 1

        except:
            pass

        # render curl counter and setup status
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
        # REPS
        cv2.putText(image, 'Reps: ', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # STAGE STATUS
        cv2.putText(image, 'STAGE: ', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # render detections
        # change colour and show landmarks on feed
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('Mediapipe Feed', image)  # visualizing webcam

        # bottom code is used to exit window/feed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()



