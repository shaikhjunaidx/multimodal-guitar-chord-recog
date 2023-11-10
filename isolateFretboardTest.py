import cv2
import mediapipe as mp
from image import Image
from guitarFunctions import *
import traceback

liveFeed = True
cameraNumber = 0
printErrors = False

# Importing required modules from mediapipe
mpDrawing = mp.solutions.drawing_utils
mpHands = mp.solutions.hands

# Specify the drawing specifications for the landmarks and connections
handLandmarkDrawingSpec = mpDrawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=4)
handConnectionDrawingSpec = mpDrawing.DrawingSpec(color=(0, 255, 0), thickness=6, circle_radius=8)

def processFrame(frame):

    # Display the original and processed frames
    cv2.imshow('Original', frame)

    # Create an Image object
    chordImage = Image(image=frame)

    # Rotate and crop the frame
    rotatedImage = rotateNeck(chordImage)
    if rotatedImage:
        croppedCoordinates = isolateNeck(rotatedImage)

        if croppedCoordinates:
            firstH, lastH, firstV, lastV = croppedCoordinates

            # Process the image using the hands object
            # Convert the image to RGB color space
            hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            results = hands.process(cv2.cvtColor(rotatedImage.image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mpDrawing.draw_landmarks(
                        image=rotatedImage.image,
                        landmark_list=hand_landmarks,
                        connections=mpHands.HAND_CONNECTIONS,
                        landmark_drawing_spec=handLandmarkDrawingSpec,
                        connection_drawing_spec=handConnectionDrawingSpec)
                    
            isolatedFretboard = rotatedImage.image[firstH - 15:lastH + 15, firstV - 15 :lastV + 15]

            isolatedNeck = Image(image=isolatedFretboard)
            if isolatedNeck:
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
               if printErrors:
                   traceback.print_exc()
            
            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()

    else:
        # Specify the path to the image you want to use
        imagePath = "capstone/17.jpg"
        frame = cv2.imread(imagePath)

        if frame is None:
            print("Error: Failed to load the image from the specified path.")
            exit(1)

        try:
            processFrame(frame)
        except Exception as e:
            if printErrors:
                traceback.print_exc()

if __name__ == '__main__':
    main()