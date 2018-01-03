#Author : Sahil Khosla 
#Copyrights made

import zbar
from PIL import Image,ImageColor
import cv2
import numpy as np

cv2.namedWindow("Python Qr-Code Detection")
cap = cv2.VideoCapture(0)

scanner = zbar.ImageScanner()
scanner.parse_config('enable')

while True:
    	ret, im = cap.read()
    	if not ret:
	  	continue 
	# Read Image
	#im = cv2.imread("sahilkhosla.jpg");
	size = im.shape
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY, dstCn=0)
    	pil = Image.fromarray(gray)
    	width, height = pil.size
    	raw = pil.tobytes()
    	image = zbar.Image(width, height, 'Y800', raw)
    	scanner.scan(image)

	for symbol in image:
        	print 'decoded', symbol.type, 'symbol', '"%s"' % symbol.data
		topLeftCorners, bottomLeftCorners, bottomRightCorners, topRightCorners = [item for item in symbol.location]
        	cv2.line(im, topLeftCorners, topRightCorners, (255,0,0),2)
       	 	cv2.line(im, topLeftCorners, bottomLeftCorners, (255,0,0),2)
        	cv2.line(im, topRightCorners, bottomRightCorners, (255,0,0),2)
        	cv2.line(im, bottomLeftCorners, bottomRightCorners, (255,0,0),2)
		#2D image points. If you change the image, you need to change vector
                
		image_points = np.array([
				            (int((topLeftCorners[0]+topRightCorners[0])/2), int((topLeftCorners[1]+bottomLeftCorners[1])/2)),     # Nose
				            topLeftCorners,     # Left eye left corner
				            topRightCorners,     # Right eye right corne
				            bottomLeftCorners,     # Left Mouth corner
				            bottomRightCorners      # Right mouth corner
				        ], dtype="double")
		 
		# 3D model points.
		model_points = np.array([
				            (0.0, 0.0, 0.0),             # Nose
				            (-225.0, 170.0, -135.0),     # Left eye left corner
				            (225.0, 170.0, -135.0),      # Right eye right corne
				            (-150.0, -150.0, -125.0),    # Left Mouth corner
				            (150.0, -150.0, -125.0)      # Right mouth corner
				         
				        ])
		 
		 
		# Camera internals
		 
		focal_length = size[1]
		center = (size[1]/2, size[0]/2)
		camera_matrix = np.array(
				         [[focal_length, 0, center[0]],
				         [0, focal_length, center[1]],
				         [0, 0, 1]], dtype = "double"
				         )
		 
		print "Camera Matrix :\n {0}".format(camera_matrix)
		 
		dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
		(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)
		 
		print "Rotation Vector:\n {0}".format(rotation_vector)
		print "Translation Vector:\n {0}".format(translation_vector)
		 
		 
		# Project a 3D point (0, 0, 1000.0) onto the image plane.
		# We use this to draw a line sticking out of the nose
		 
		 
		(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 100.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
		 
		for p in image_points:
		    cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
		 
		 
		p1 = ( int(image_points[0][0]), int(image_points[0][1]))
		p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
		
                #pillars 
		cv2.line(im, bottomLeftCorners, p2, (255,0,0), 2)
		cv2.line(im, topLeftCorners, p2, (255,0,0), 2)
		cv2.line(im, bottomRightCorners, p2, (255,0,0), 2)
		cv2.line(im, topRightCorners, p2, (255,0,0), 2)
		 
		# Display image
	cv2.imshow("Output", im)
	
	# Wait for the magic key
    	keypress = cv2.waitKey(1) & 0xFF
    	if keypress == ord('q'):
		break

cv2.waitKey(0)
