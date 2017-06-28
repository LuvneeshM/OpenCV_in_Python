import cv2
import sys

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

#use default webcam
videoCapture = cv2.VideoCapture(0)

while True: 
	#returns: 
	#actual video frame (one frame per loop)
	#return code
	ret, frame = videoCapture.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor = 1.2,
		minNeighbors = 5,
		minSize = (30,30),
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

videoCapture.release()
cv2.destroyAllWindows()