Clone the repo and run isolateFretboardTest.py
Make sure to choose the correct camera number in the beginning of the file so that it works with the intended external camera

Currently, three windows are displayed. The original image, the hough grid and the isolated fretboard.

Hough Transform is sensitive to nois and change in lighting conditions. Move your fretboard such that maximum lines (both horizontal and vertical) can be detected.

My algorithm assumes that the fretboard is well lit and that the bridge (white plate at the beginning of the guitar) is always in the frame at the beginning of the frame.

Currently my program does not detect hand landmarks as they tend to slow it down. I am working on optimizing it now.