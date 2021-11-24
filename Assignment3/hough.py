import matplotlib.pyplot as plt
import numpy as np

def getCrossLine(y1, y2, x):
    idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
    theta = x[idx]
    r = y1[idx]
    ixa = np.arange(start=0, stop=7, step=0.5)
    iya = np.zeros(len(ixa))
    for i in range(len(ixa)):
        iya[i] = ((-np.cos(theta)/np.sin(theta))*ixa[i]) + (r/np.sin(theta))
    return ixa, iya

thetas = np.deg2rad(np.arange(-90, 90))

xa = [2, 3, 6]
ya = [2, 1.5, 0]

ra = []
for i in range(len(xa)):
    temp = []
    for k in range(len(thetas)):
        temp.append(xa[i]*np.cos(thetas[k]) + ya[i] * np.sin(thetas[k]))
    ra.append(temp)


xc = [2, 5, 6]
yc = [2, 3, 0]
rc = []
for i in range(len(xc)):
    temp = []
    for k in range(len(thetas)):
        temp.append(xc[i]*np.cos(thetas[k]) + yc[i] * np.sin(thetas[k]))
    rc.append(temp)

ra = np.asarray(ra)
rc = np.asarray(rc)

ixa, iya = getCrossLine(ra[0], ra[1], thetas)

ixc1, iyc1 = getCrossLine(rc[0], rc[1], thetas)
ixc2, iyc2 = getCrossLine(rc[1], rc[2], thetas)
ixc3, iyc3 = getCrossLine(rc[0], rc[2], thetas)



fig , (ax1, ax2) = plt.subplots(1,2)
fig.suptitle('A')
ax1.scatter(xa, ya)
ax1.plot(ixa, iya)
for r in ra:
    ax2.plot(thetas, r)


fig2 , (cx1, cx2) = plt.subplots(1,2)
fig2.suptitle('C')
cx1.scatter(xc, yc)
cx1.plot(ixc1, iyc1)
cx1.plot(ixc2, iyc2)
cx1.plot(ixc3, iyc3)
for r in rc:
    cx2.plot(thetas, r)
plt.show()
