import cv2
import time
import imutils
from imutils.object_detection import non_max_suppression


subject_label = 1
font = cv2.FONT_HERSHEY_SIMPLEX
list_of_videos = []
cascade_path = "face_cascades/haarcascade_profileface.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
recognizer = cv2.face.LBPHFaceRecognizer_create()
count=0

def detect_people(frame):
	"""
	detect humans using HOG descriptor
	Args:
		frame:
	Returns:
		processed frame
	"""
	(rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.06)
	rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
	return frame



def detect_face(frame):
	"""
	detect human faces in image using haar-cascade
	Args:
		frame:
	Returns:
	coordinates of detected faces
	"""
	faces = face_cascade.detectMultiScale(frame, 1.1, 2, 0, (20, 20) )
	return faces


def draw_faces(frame, faces):
	"""
	draw rectangle around detected faces
	Args:
		frame:
		faces:
	Returns:
	face drawn processed frame
	"""
	for (x, y, w, h) in faces:
		xA = x
		yA = y
		xB = x + w
		yB = y + h
		cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
	return frame


def put_label_on_face(frame, faces, labels):
	"""
	draw label on faces
	Args:
		frame:
		faces:
		labels:
	Returns:
		processed frame
	"""
	i = 0
	for x, y, w, h in faces:
		cv2.putText(frame, str(labels[i]), (x, y), font, 1, (255, 255, 255), 2)
		i += 1
	return frame

def background_subtraction(previous_frame, frame_resized_grayscale, min_area):
	"""
	This function returns 1 for the frames in which the area
	after subtraction with previous frame is greater than minimum area
	defined.
	Thus expensive computation of human detection face detection
	and face recognition is not done on all the frames.
	Only the frames undergoing significant amount of change (which is controlled min_area)
	are processed for detection.
	"""
	frameDelta = cv2.absdiff(previous_frame, frame_resized_grayscale)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	temp=0
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) > min_area:
			temp=1
	return temp


if __name__ == '__main__':
	"""
	main function
	"""

	camera = cv2.VideoCapture(0)
	time.sleep(0.25)
	grabbed, frame = camera.read()
	print(frame.shape)
	frame_resized = imutils.resize(frame, width=min(800, frame.shape[1]))
	frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
	print(frame_resized.shape)

	# defining min cuoff area
	min_area = (3000 / 800) * frame_resized.shape[1]

	while True:
		starttime = time.time()
		previous_frame = frame_resized_grayscale
		grabbed, frame = camera.read()
		if not grabbed:
			break
		frame_resized = imutils.resize(frame, width=min(800, frame.shape[1]))
		frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
		temp = background_subtraction(previous_frame, frame_resized_grayscale, min_area)
		if temp == 1:
			frame_processed = detect_people(frame_resized)
			faces = detect_face(frame_resized_grayscale)
			if len(faces) > 0:
				frame_processed = draw_faces(frame_processed, faces)

			cv2.imshow("Detected Human and face", frame_processed)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
			endtime = time.time()
			print("Time to process a frame: " + str(starttime - endtime))
		else:
			count = count + 1
			print("Number of frame skipped in the video= " + str(count))

	camera.release()
	cv2.destroyAllWindows()