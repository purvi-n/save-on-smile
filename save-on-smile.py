import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
smile_cascade = cv.CascadeClassifier('haarcascade_smile.xml')

cap = cv.VideoCapture(0)	

while True:
	_, img = cap.read()
	pic = img

	grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(grayimg, 1.1, 4)
	smiles = smile_cascade.detectMultiScale(grayimg, 3, 20)

	for x, y, w, h in faces:
		cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

	for x, y, w, h in smiles:
		cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
		cv.imwrite('pic.png', pic)

	cv.imshow('img', img)
	esc = cv.waitKey(30) & 0xff
	if esc == 27:
		break

cap.release()