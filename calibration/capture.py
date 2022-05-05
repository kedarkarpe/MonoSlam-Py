import cv2

cap = cv2.VideoCapture(0)

success = True
count = 0


while cap.isOpened():
	print(count)

	success, img = cap.read()
	if success:
		cv2.imshow('img', img)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

		if cv2.waitKey(25) & 0xFF == ord('0'):
			cv2.imwrite('img/frame%d.jpg' % count, img)
			count += 1

	if count  == 2:
		break