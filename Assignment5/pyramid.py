import numpy as np
import cv2
import os

frame1 = cv2.imread("im1.jpg")
frame2 = cv2.imread("im2.jpg")

frame1g = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame2g = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

combined = cv2.addWeighted(frame1, 0.3, frame2, 0.5, 0)
cv2.imwrite("pyramid/combined.png", combined)

parameters = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

p0 = cv2.goodFeaturesToTrack(frame1g, mask=None, **parameters)

maxLevels = [2,3]
maxIterations = [2, 10]

for level in maxLevels:
    print("On Max Level: ", level)
    for iter in maxIterations:
        OFparams = dict(
            winSize=(15, 15),
            maxLevel=level,
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_COUNT, iter, 0.03),
        )
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            frame1g, frame2g, p0, None, **OFparams)
        good_new = p1[st == 1]
        good_prev = p0[st == 1]

        arrows = np.zeros_like(combined)

        for i, (n, p) in enumerate(zip(good_new, good_prev)):
            nx, ny = n.ravel()
            px, py = p.ravel()

            nx = int(nx)
            ny = int(ny)
            px = int(px)
            py = int(py)

            arrows = cv2.arrowedLine(arrows, (px, py), (nx, ny), [
                                     255, 0, 255], 2, cv2.LINE_AA, tipLength=0.4)

            output = cv2.add(combined, arrows)
            if(not os.path.exists("pyramid/"+str(level))):
                os.mkdir("pyramid/"+str(level))
            outName = "pyramid/" + str(level)+"/" + \
                str(level) + "levels-" + str(iter)+"iterations.png"
            cv2.imwrite(outName, output)
