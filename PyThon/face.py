import numpy as np
import argparse
import cv2

frame = cv2.imread("i.jpg")
 
# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
cv2.imwrite("converted.png", converted)
        #
skinMask = cv2.inRange(converted, lower, upper)

	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        #
skinMask = cv2.erode(skinMask, kernel, iterations = 2)
cv2.imwrite("erode.png", skinMask)
        
skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
cv2.imwrite("dilate.png", skinMask)
        #
	# blur the mask to help remove noise, then apply the
	# mask to the frame
skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
cv2.imwrite("skinMask.png", skinMask)
        #
skin = cv2.bitwise_and(frame, frame, mask = skinMask)
cv2.imwrite("masked.png", skin)
        # show the skin in the image along with the mask
        #cv2.imshow("images", np.hstack([frame, skin]))
        # if the 'q' key is pressed, stop the loop
# if cv2.waitKey(1) & 0xFF == ord("q"):
# 	exit

# cleanup the camera and close any open windows

# cv2.destroyAllWindows()
