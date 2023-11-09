import cv2
import mediapipe as mp
from image import Image
from guitarFunctions import *

liveFeed = True
cameraNumber = 0

# Importing required modules from mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Specify the drawing specifications for the landmarks and connections
hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=6)
hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=4, circle_radius=10)

def processFrame(frame):

    # Need to work on this, how can I make it efficient and fast?

    # Process the image using the hands object
    # Convert the image to RGB color space
    # hands = mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1)
    # results_hand = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


    # Draw the hand landmarks and connections if any are detected

    # if results_hand.multi_hand_landmarks:
    #     for hand_landmarks in results_hand.multi_hand_landmarks:
    #         mp_drawing.draw_landmarks(
    #             image=frame,
    #             landmark_list=hand_landmarks,
    #             connections=mp_hands.HAND_CONNECTIONS,
    #             landmark_drawing_spec=hand_landmark_drawing_spec,
    #             connection_drawing_spec=hand_connection_drawing_spec)

    # Create an Image object
    chordImage = Image(image=frame)

    # Rotate and crop the frame
    rotatedImage = rotateNeck(chordImage)
    if rotatedImage:
        isolatedNeck = isolateNeck(rotatedImage)


    # Display the original and processed frames
    cv2.imshow('Original', frame)

    if isolatedNeck is not None:
        cv2.imshow('Cropped neck image', isolatedNeck.image)


def main():
    # Create a hands object from mediapipe
    if liveFeed:
        cap = cv2.VideoCapture(cameraNumber)
        while True:
            # Capture a frame from the camera
            ret, frame = cap.read()

            if not ret:
                break

            try:
                processFrame(frame)
            except Exception as e:
                print(str(e))
            
            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()

    else:
        # Specify the path to the image you want to use
        image_path = "capstone/17.jpg"
        frame = cv2.imread(image_path)

        if frame is None:
            print("Error: Failed to load the image from the specified path.")
            exit(1)

        try:
            processFrame(frame)
        except Exception as e:
            print(str(e))

if __name__ == '__main__':
    main()