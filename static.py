import numpy as np
import cv2
import matplotlib.pyplot as plt
cv2.namedWindow("frame mask",cv2.WINDOW_NORMAL)
cv2.namedWindow("Frame_final",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_dilate",cv2.WINDOW_NORMAL)

vid_path = "demo1.mp4"
cap = cv2.VideoCapture(vid_path)
backSub = cv2.createBackgroundSubtractorMOG2(history = 30,varThreshold = 32,detectShadows=True)
backSub.setShadowThreshold(0.5)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.GaussianBlur(frame, (3, 3), 0)
    
    # Apply background subtraction
    fg_mask = backSub.apply(frame,learningRate = -1)
    print("frame",fg_mask)
    # Find contours
    cv2.imshow('frame mask', fg_mask)

    # apply global threshold to remove shadows
    retval, mask_thresh = cv2.threshold( fg_mask, 100, 255, cv2.THRESH_BINARY)

    # set the kernal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_dia = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # Apply erosion
    mask_eroded = cv2.erode(mask_thresh, kernel, iterations=1)
    mask_dilate = cv2.dilate(mask_eroded, kernel_dia, iterations=1)
    cv2.imshow('mask_dilate', mask_dilate)
    contours, hierarchy = cv2.findContours(mask_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 50  # Define your minimum area threshold
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    frame_out = frame.copy()
    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
    
    # Display the resulting frame
    cv2.imshow('Frame_final', frame_out)
    cv2.waitKey(10)
    