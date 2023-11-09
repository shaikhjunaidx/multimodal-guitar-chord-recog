import cv2
import mediapipe as mp
import numpy as np

from typing import *
from image import Image
from copy import deepcopy
from collections import defaultdict
from math import inf
from statistics import median

def binarize(image, thresholdValue):
    thisImage = image
    thisImage[thisImage <= thresholdValue] = 0
    thisImage[thisImage > thresholdValue] = 255
    return thisImage


def rotate(imageArray, angle, center=None, scale=1.0):
    (height, width) = imageArray.shape[:2]

    if center is None:
        center = (width / 2, height / 2)

    rotationMatrix = cv2.getRotationMatrix2D(center, angle, scale)

    return cv2.warpAffine(imageArray, rotationMatrix, (width, height))


def rotateNeck(image):
    imageArray = image.image
    # make a deep copy of the original image arary so that we can draw hough lines on it and display
    houghImage = deepcopy(image.image)

    edges = image.sobelY()

    edges = binarize(edges, 150)

    lines = image.probHoughTransform(edges, 30, 50)  # TODO: Calibrate params automatically
    slopes = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # calculate slope
                slopes.append(abs((y2 - y1) / (x2 - x1)))
                # draw the hough lines detected
                cv2.line(houghImage, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Draw lines in red

        medianSlope = median(slopes)
        angle = medianSlope * 55
    else:
        angle = 0

    return Image(image=rotate(imageArray, -angle))


def isolateNeck(image):

    # Expected fretboard dimensions
    expectedFretboardWidth, expectedFretboardHeight = 600, 100
    minFretboardWidth, minFretboardHeight = 500, 90
    maxFretboardWidth, maxFretboardHeight = 650, 120
    fretboardThreshold = 40  

    # if image is empty, return
    if not image:
        return

    # make a deep copy of the image array that will be cropped, we will later use this to make a new Image object
    cropThisImageArr = deepcopy(image.image)

    # cv2.imshow("Rotated image", image_to_crop)

    # image on which we will draw the horizontal lines
    horizontalLinesImage = deepcopy(image.image)
    # image on which we will draw the vertical lines
    verticalLinesImage = deepcopy(image.image)
    # image on which we will draw both horizontal and vertical lines
    gridImage = deepcopy(image.image)


    # -------DETECTING HORIZONTAL LINES ALONG THE STRINGS------- #

    # detecting sobelY edges on the input image
    stringEdges = image.sobelY()
    stringEdges = binarize(stringEdges, 100)

    stringHoughLines = image.probHoughTransform(stringEdges, 50, 50)

    horizontalLines = []
    horizontalSlopeThreshold = 10
    horizontalDiffThreshold = 6

    if stringHoughLines is None:
        return None

    for line in stringHoughLines:
        for x1, y1, x2, y2 in line:
            # check if the line is straight or not
            # i.e., (1, 1) ----- (1, 1) is a straight line
            # since the lines will not always be straight, check against a threshold
            if abs(y2 - y1) < horizontalSlopeThreshold: 
                horizontalLines.append(y1)
                horizontalLines.append(y2)
                cv2.line(horizontalLinesImage, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.line(gridImage, (x1, y1), (x2, y2), (0, 0, 255), 2)

    sortedHorizontalLines = list(sorted(horizontalLines))
    horizontalDiff = [0]

    firstH = 0
    lastH = inf

    for i in range(len(sortedHorizontalLines) - 1):
        horizontalDiff.append(sortedHorizontalLines[i + 1] - sortedHorizontalLines[i])

    for i in range(len(horizontalDiff) - 1):
        if horizontalDiff[i] < horizontalDiffThreshold:
            lastH = sortedHorizontalLines[i]
            if i > 3 and firstH == 0:
                firstH = sortedHorizontalLines[i]


    # -------DETECTING VERTICAL LINES ALONG THE FRETS------- #


    fretEdges = image.sobelX()
    fretEdges = binarize(fretEdges, 100)

    verticalKernel = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ], dtype=np.uint8)

    # Perform dilation to close gaps between edges
    dilatedFrets = cv2.dilate(fretEdges, verticalKernel, iterations=1)

    # Perform erosion to restore the original size while keeping the gaps closed
    closedFrets = cv2.erode(dilatedFrets, verticalKernel, iterations=1)

    fretEdges = closedFrets

    fretHoughLines = image.probHoughTransform(fretEdges, 50, 50)

    verticalLines = []
    verticalSlopeThreshold = 5
    verticalDiffThreshold = 10

    if fretHoughLines is None:
        return None

    for line in fretHoughLines:
        for x1, y1, x2, y2 in line:
            if abs(x1 - x2) < verticalSlopeThreshold:
                verticalLines.append(x1)
                verticalLines.append(x2)
                cv2.line(verticalLinesImage, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.line(gridImage, (x1, y1), (x2, y2), (0, 255, 0), 1)


    sortedVerticalLines = list(sorted(verticalLines))
    verticalDiff = [0]

    firstV = 0
    lastV = inf

    for i in range(len(sortedVerticalLines) - 1):
        verticalDiff.append(sortedVerticalLines[i + 1] - sortedVerticalLines[i])

    for i in range(len(verticalDiff) - 1):
        if verticalDiff[i] < verticalDiffThreshold:
            lastV = sortedVerticalLines[i]
            if i > 2 and firstV == 0:
                firstV = sortedVerticalLines[i]

    

    cv2.imshow("Image with vertical and horizontal hough lines", gridImage)

    # get the last V and H coordinate (X and Y)
    if lastV == inf:
        lastV = len(cropThisImageArr) - 1
    if lastH == inf:
        lastH = len(cropThisImageArr) - 1

    # Calculate the width and height of the detected fretboard
    detectedFretboardWidth, detectedFretboardHeight = lastV - firstV, lastH - firstH
    
    # Check if the detected size is within the expected range
    if (abs(detectedFretboardWidth - expectedFretboardWidth) <= fretboardThreshold 
        and abs(detectedFretboardHeight - expectedFretboardHeight) <= fretboardThreshold):
        
        # print("----------")
        # print("detectedFretboardWidth", detectedFretboardWidth)
        # print("expectedFretboardHeight", detectedFretboardHeight)
        # print("----------")

        # Coordinates are close to the expected size, proceed with cropping
        isolatedFretboard = cropThisImageArr[firstH - 15:lastH + 15, firstV :lastV + 15] 

        # return the cropped Image object only if it is a valid fretboard crop and the bridge (white part at start of the board)
        # is present in this crop
        if (bridgePresent(isolatedFretboard) 
            and maxFretboardHeight > len(isolatedFretboard) > minFretboardHeight 
            and maxFretboardWidth > len(isolatedFretboard[0]) > minFretboardWidth):
            return Image(image=isolatedFretboard)
        
    # Detected size is not close to expected, return None
    return None


def bridgePresent(board):

    # Convert the image to grayscale
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)

    # Define a threshold to determine what is considered as white
    whitePixels = 1300 

    # Extract the rightmost region of interest (ROI) with a small width
    rightmostROI = gray[:, -20:]

    binary = binarize(rightmostROI, 127)

    # Count the number of white pixels in the rightmost ROI
    white_pixel_count = cv2.countNonZero(binary)

    # Check if there are enough white pixels to consider the white rectangle (bridge) as present
    return white_pixel_count > whitePixels

