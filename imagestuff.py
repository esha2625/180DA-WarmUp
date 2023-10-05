#CITATIONS
#Threshold: https://stackoverflow.com/questions/26218280/thresholding-rgb-image-in-opencv
#Contours: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
#Dominant Color Processing: https://stackoverflow.com/questions/73808864/get-most-dominant-colors-from-video-opencv-python
#Dominant Color Processing Image: https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097


import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'

cap = cv2.VideoCapture(0)

cv2.namedWindow(window_capture_name)
cv2.namedWindow(window_detection_name)
n_clusters = 1
while(True):
    # # Capture frame-by-frame
    # ret, frame = cap.read()

    # # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # frame_threshold = cv2.inRange(frame_HSV, (100, 0, 0), (132, 255, 255))
    # contours,hierarchy = cv2.findContours(frame_threshold, 1, 2)
    # for cnt in contours:
    #      rect = cv2.minAreaRect(cnt)
    #      box = cv2.boxPoints(rect)
    #      box = np.intp(box)
    #     #  cv2.drawContours(frame, [box], 0, (0,0,0), 2)

    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    # clt = KMeans(n_clusters=3) #cluster number
    # clt.fit(img)
    
    # # Display the resulting frame
    # cv2.imshow(window_capture_name, frame)
    # cv2.imshow(window_detection_name, frame_threshold)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    status, image = cap.read()
    if not status:
        break

    # to reduce complexity resize the image
    data = cv2.resize(image, (100, 100)).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data.astype(np.float32), 3, None, criteria, 10, flags)

    cluster_sizes = np.bincount(labels.flatten())

    palette = []
    for cluster_idx in np.argsort(-cluster_sizes):
        palette.append(np.full((image.shape[0], image.shape[1], 3), fill_value=centers[cluster_idx].astype(int), dtype=np.uint8))
    palette = np.hstack(palette)

    sf = image.shape[1] / palette.shape[1]
    out = np.vstack([image, cv2.resize(palette, (0, 0), fx=sf, fy=sf)])

    cv2.imshow("dominant_colors", out)
    cv2.waitKey(1)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
