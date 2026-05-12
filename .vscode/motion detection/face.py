# pip install opencv-python

import cv2  # Fixed capitalization

# Load the cascades
face_cascade = cv2.CascadeClassifier('D:/xmll f/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('D:/xmll f/haarcascade_eye.xml')
cap = cv2.VideoCapture('D:/xmll f/WALK.mp4')


while True:
    ret, img = cap.read()
    
    # Safety Check: If the frame didn't load (end of video or wrong path), break the loop
    if not ret:
        print("Done processing or file not found.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 3)

    cv2.imshow('Face & Eye Detection', img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
