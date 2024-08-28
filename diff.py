import cv2
import numpy as np
import time

cv2.namedWindow("feed",cv2.WINDOW_NORMAL)
cv2.namedWindow("diff",cv2.WINDOW_NORMAL)
cv2.namedWindow("gray",cv2.WINDOW_NORMAL)
cv2.namedWindow("thresh",cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture('cars.avi')
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

# out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280,720))

ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)
while cap.isOpened():
    t0 = time.time()
    diff = cv2.absdiff(frame1, frame2)
    print("dfii", diff)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    blur = cv2.GaussianBlur(gray, (1,1), 0)
    cv2.imshow("diff", blur)
    _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh", thresh)
    kernel = np.ones((5, 5), np.uint8) 
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 50 or w/h > 1.7 or h/w > 1.7:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    image = cv2.resize(frame1, (1280,720))
    # out.write(image)
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
    print("time",time.time()-t0)
    if cv2.waitKey(30) == 27:
        break

cv2.destroyAllWindows()
cap.release()
# out.release()