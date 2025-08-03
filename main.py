import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
from sklearn.model_selection import train_test_split
from utils import (
    mp_holistic, mp_drawing, DATA_PATH, actions, no_sequences, 
    sequence_length, label_map, mediapipe_detection, draw_landmarks, 
    extract_keypoints, load_training_data, create_data_directories
)

# Load training data
X, y = load_training_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)




def main():
    # Create necessary directories
    create_data_directories()

    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:

        # NEW LOOP
        # Loop through actions
        for action in actions:
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):

                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_landmarks(image, results)

                    # NEW Apply wait logic
                    if frame_num == 0:
                        cv2.putText(
                            image,
                            "STARTING COLLECTION",
                            (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            4,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            image,
                            "Collecting frames for {} Video Number {}".format(
                                action, sequence
                            ),
                            (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        # Show to screen
                        cv2.imshow("OpenCV Feed", image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(
                            image,
                            "Collecting frames for {} Video Number {}".format(
                                action, sequence
                            ),
                            (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        # Show to screen
                        cv2.imshow("OpenCV Feed", image)

                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(
                        DATA_PATH, action, str(sequence), str(frame_num)
                    )
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
