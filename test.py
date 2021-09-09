# import os
# import cv2 as cv
# import numpy as np
# #make an array of 120000 random bytes
# randombytearray=bytearray(os.urandom (120000))
# #translate into numpy array
# flatnumpyarray=np.array(randombytearray)
# #convert the array to make a 400 * 300 grayscale image
# grayimage=flatnumpyarray.reshape(300, 400)
# #show gray image
# cv.imshow("grayimage", grayimage)
# #print image "s array
# print(grayimage)
# cv.waitKey()
# #byte array translate into rgb image
# randombytearray1=bytearray(os.urandom (360000))
# flatnumpyarray1=np.array(randombytearray1)
# bgrimage=flatnumpyarray1.reshape(300,400,3)
# cv.imshow("bgrimage", bgrimage)
# cv.waitKey()
# cv.destroyAllWindows()


# import numpy as np
# a = None
# a = np.zeros((9,2,3), dtype=int)

# print(a.shape[1])

# for i in np.nditer(a,op_flags = ['readwrite']):
#     i[...] = 2.0

# for i in range(a.shape[0]):
#     for j in range(a.shape[1]):
#         for k in range(a.shape[2]):
#             if(i == 4):
#                 a[i][j][k] = 3
#             print(a[i][j][k])


# print(a)

import numpy as np
import cv2
import math

# def dist(a,b,c,x,y):
#     return (a*x + b*y + c)/math.sqrt(a**2 + b**2)


# r = 600
# c = 800


# data = np.ones((r,c), np.uint8)


# maxdist = dist(c, -r, r*c, 0, 0)

# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         data[i][j] = 128-int(dist(c, -r, r*c, -i, j)*127/maxdist)

# img = cv2.applyColorMap(cv2.convertScaleAbs(data, alpha=1.0, beta=0.0), cv2.COLORMAP_BONE)

# cv2.imshow("img", img)

# while True:
#     key = cv2.waitKey(1)
#     if key == 27:
#         cv2.destroyAllWindows()
#         break

# print(data)


# a = np.ones((4,5), np.int8)*6
# b = np.ones((4,5), np.float32)*2

# c = np.subtract(a,b)
# print(a)
# print(c)

# for i in range(-5//2,5//2):
#     print (10 - i)

# l =  list()
# l.append((9,1))
# l.append((3,1))
# l.append((9,2))
# print(l)



# def binsearch(val , ls):
#     n = len(ls)
#     if(n == 0) or (n == 1 and val!=ls[0]):
#         return False
#     elif (val < ls[n//2]):
#         return binsearch(val, ls[0:(n//2) -1])
#     elif (val > ls[n//2]):
#         return binsearch(val, ls[(n//2)-1:-1])
#     else:
#         return True


# ls = list(map(int,input().split(' ')))

# ls = sorted(ls)

# print(binsearch(1,ls))


import random

radius = 400
rangeX = (0, 2500)
rangeY = (0, 2500)
qty = 10  # or however many points you want

# Generate a set of all points within 200 of the origin, to be used as offsets later
# There's probably a more efficient way to do this.
deltas = set()
for x in range(-radius, radius+1):
    for y in range(-radius, radius+1):
        if x*x + y*y <= radius*radius:
            deltas.add((x,y))

randPoints = []
excluded = set()
i = 0
while i<qty:
    x = random.randrange(*rangeX)
    y = random.randrange(*rangeY)
    if (x,y) in excluded: continue
    randPoints.append((x,y))
    i += 1
    excluded.update((x+dx, y+dy) for (dx,dy) in deltas)
print(randPoints)
