import cv2
import mediapipe as mp
from image import Image
from guitarFunctions import *
import traceback
import os

cameraNumber = 0
printErrors = False

# Importing required modules from mediapipe
mpDrawing = mp.solutions.drawing_utils
mpHands = mp.solutions.hands

# Specify the drawing specifications for the landmarks and connections
handLandmarkDrawingSpec = mpDrawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=4)
handConnectionDrawingSpec = mpDrawing.DrawingSpec(color=(0, 255, 0), thickness=6, circle_radius=8)

# Function to create a folder
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def processFrame(frame):

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
                    return isolatedNeck


def main():

    numberOfImages = 600
    thisChord = "F"

    # Get the current directory of the Python script
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Create a folder for the current chord
    currentChordDirectory = os.path.join(current_directory, "chords", thisChord)
    create_folder(currentChordDirectory)


    # Initialize variablesQA
    imageNumber = 0
    startCapturing = False
    frames = 0

    cap = cv2.VideoCapture(cameraNumber)
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

        try:
            isolatedFretBoard = processFrame(frame)
            if isolatedFretBoard:
                saveThis = isolatedFretBoard.image
                if len(saveThis) < 90 or len(saveThis[0]) < 500:
                    continue
                # Display a message indicating that images are being captured
                cv2.putText(frame, "Capturing...", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (127, 255, 255), 5)

                # If image capturing is turned on
                if startCapturing:
                    # Save the current image
                    cv2.imwrite(currentChordDirectory + "/" + str(imageNumber) + ".jpg", saveThis)
                    frames += 1
                    imageNumber += 1
                
                # Display the current image number
                cv2.putText(frame, str(imageNumber), (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (127, 127, 255), 3)

                cv2.imshow('Cropped neck image', isolatedFretBoard.image)
                # Display the original and processed frames
                cv2.imshow('Original', frame)

            # Exit the loop if the 'q' key is pressed

        except Exception as e:
            if printErrors:
                traceback.print_exc()
        
        keypress = cv2.waitKey(1)

        # Exit the loop if the 'q' key is pressed
        if keypress == ord('q'):
            break

        if keypress == ord('c'):
            # Toggle image capturing on/off
            if not startCapturing:
                startCapturing = True
            else:
                startCapturing = False
                frames = 0

        # If the required number of images is captured, break the loop
        if imageNumber == numberOfImages:
            break
    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()