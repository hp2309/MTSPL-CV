from os import defpath
from typing import OrderedDict
import numpy as np
import pyrealsense2 as rs
import cv2
import argparse
import os.path
import matplotlib.pyplot as plt
import math

parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.Remember to change the stream fps and format to match the recorded.")
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
args = parser.parse_args()

if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()

pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, args.input)
pipeline.start(config)

colorizer = rs.colorizer(float(2))

def nothing(x):
    pass

def dist(a,b,c,x,y):
    return (a*x + b*y + c)/math.sqrt(a**2 + b**2)

def correctionMatrix(data, alpha, beta):
    r = data.shape[0]
    c = data.shape[1]
    maxdist = dist(c, -r, r*c, 0, 0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = (dist(c, -r, r*c, -i, j)*alpha/maxdist)+beta
    
    return data

def find_depth(points, ground_depth, depth_data, binary, k_size, skip_freq, slice_ratio):
    av_thickness = list()
    med_thickness = list()
    avg_w_from_ground = list()
    avg_b_from_ground = list()


    for i in range(0, len(points)):
        if i % skip_freq == 0:
            white_list = list()
            black_list = list()
            point = points[i][0]
            x = point[1]
            y = point[0]
            for j in range(-1*(k_size//2), (k_size//2)):
                for k in range(-1*(k_size//2), (k_size//2)):
                    # print(x,y,i,j,k)
                    if(binary[x+j][y+k] == 255):
                        white_list.append(depth_data[x+j][y+k])
                    else:
                        black_list.append(depth_data[x+j][y+k])
    
            white_list.sort()
            black_list.sort()
            new_white_list = white_list[int(-len(white_list)/slice_ratio)::1]
            new_black_list = black_list[0::int(len(black_list)/slice_ratio)]
            #mean
            sum = 0
            count = 0
            for val in new_white_list:
                sum += val
                count+=1
            avg_w = sum/count
            avg_w_from_ground.append(ground_depth-avg_w)


            sum = 0
            count = 0
            for val in new_black_list:
                sum += val
                count+=1
            avg_b = sum/count
            avg_b_from_ground.append(ground_depth-avg_b)

            avg_thickness = avg_b - avg_w

            av_thickness.append(avg_thickness)

            med_b = black_list[len(black_list)//2]
            med_w = white_list[len(white_list)//2]
            # print(white_list, " <|> ", black_list)
            med_thickness.append(med_b-med_w)


    av_thickness.sort()

    sum = 0
    c = 0
    for v in av_thickness:
        sum += v
        c+=1
    avg_th2 = sum/c

    sum = 0
    c = 0
    for v in med_thickness:
        sum += v
        c+=1
    med_th = sum/c

    sum = 0
    c = 0
    for v in avg_w_from_ground:
        sum += v
        c+=1
    avg_th3 = sum/c

    sum = 0
    c = 0
    for v in avg_b_from_ground:
        sum += v
        c+=1
    avg_th4 = sum/c

    return avg_th2, med_th, avg_th3, avg_th4


def find_depth2(edge, depth_data, ground_depth, slice_ratio):
    depths = list()
    for i in range(len(edge)):
        point = edge[i][0]
        x = point[1]
        y = point[0]
        depths.append(depth_data[x][y])
    depths.sort()
    
    depths = depths[-len(depths)//slice_ratio::1]

    sum = 0
    c = 0
    for v in depths:
        sum+=v
        c+=1
    avg_depth = sum/c

    return ground_depth - avg_depth


cv2.namedWindow("sliders", cv2.WINDOW_FREERATIO)
cv2.resizeWindow("sliders", 800,800)
cv2.createTrackbar("canny_thr1_1", "sliders",29,255,nothing)
cv2.createTrackbar("canny_thr2_1", "sliders",42,255,nothing)
cv2.createTrackbar("mediankSize", "sliders", 5, 50, nothing)
cv2.createTrackbar("alp", "sliders",100,150,nothing)
cv2.createTrackbar("beta", "sliders",0,255,nothing)
cv2.createTrackbar("thr", "sliders",65,255,nothing)
cv2.createTrackbar("scale", "sliders",20,100,nothing)
cv2.createTrackbar("canny_thr1_2", "sliders",29,255,nothing)
cv2.createTrackbar("canny_thr2_2", "sliders",42,255,nothing)
cv2.createTrackbar("canny_thr1_3", "sliders",29,255,nothing)
cv2.createTrackbar("canny_thr2_3", "sliders",42,255,nothing)

# 5 frames for camera adjustment
for i in range(5):
    frame = pipeline.wait_for_frames()

frames = []
for x in range(10):
    frameset = pipeline.wait_for_frames()
    frames.append(frameset.get_depth_frame())

pipeline.stop()
print("Frames Captured")
preprocessed = np.asanyarray(colorizer.colorize(frames[0]).get_data())
preprocessed = cv2.resize(preprocessed,(425,240))

cv2.imshow("before pre-processing", preprocessed)

decimation = rs.decimation_filter()
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)
for x in range(10):
    frame = frames[x]
    frame = decimation.process(frame)
    frame = depth_to_disparity.process(frame)
    frame = spatial.process(frame)
    frame = temporal.process(frame)
    frame = disparity_to_depth.process(frame)
    frame = hole_filling.process(frame)

colorized_depth = np.asanyarray(colorizer.colorize(frame).get_data())

print(f"type of frame: {type(frames[0])}")

cv2.imshow("after pre-processing", colorized_depth)


depthf = frame

depthd = np.asanyarray(depthf.get_data())
depthd = depthd[10:230, 30:420]
ground = depthd[50:80, 340:370]
ground_depth = np.average(ground)

depthc = np.asanyarray(colorizer.colorize(depthf).get_data())
cv2.imshow("Colorized Depth Frame", depthc)

min = 1000
max = 1
originaldepth = np.copy(depthd)
for x in np.nditer(depthd, op_flags=['readonly']):
    if x<min and x!=0:
        min = int(x)
    if x>max:
        max = int(x)

delta  = max-min
for i in range(depthd.shape[0]):
    for j in range(depthd.shape[1]):
        depthd[i][j] = int(254*((max-int(depthd[i][j]))/delta))+1

correctionmatrix = correctionMatrix(np.ones(depthd.shape, np.float32), 1, 0)

flag = False

print(">> Ground Depth : " + str(ground_depth))

measurement_scale = 655/221

while True:
    scale = cv2.getTrackbarPos("scale", "sliders")
    newdepthd = np.subtract(depthd,correctionmatrix*scale)
    cv2.imshow("Correction Matrix",correctionmatrix)
    alp = float(cv2.getTrackbarPos("alp", "sliders")/100.0)
    beta = float(cv2.getTrackbarPos("beta", "sliders")*(-1))
    t1_1 = cv2.getTrackbarPos("canny_thr1_1", "sliders")
    t2_1 = cv2.getTrackbarPos("canny_thr2_1", "sliders")
    ks = cv2.getTrackbarPos("mediankSize", "sliders")*2 +1
    
    
    
    gray = cv2.cvtColor(cv2.applyColorMap(cv2.convertScaleAbs(newdepthd, alpha=alp, beta=beta), cv2.COLORMAP_BONE), cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, ks)
    canny = cv2.Canny(gray, t1_1, t2_1)
    
    
    source = gray
    m = cv2.getTrackbarPos("thr", "sliders")
    (thresh, Bin) = cv2.threshold(source, m, 255, cv2.THRESH_BINARY)
    cv2.imshow("Binary",Bin)
    (cnts, _) = cv2.findContours(Bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    index = 0
    if (len(cnts) != 0):
        largestContour = cnts[0]
        flag = True
    
    if (flag):
        for i in range(1,len(cnts)):
            if (cv2.contourArea(largestContour) < cv2.contourArea(cnts[i])):
                index = i
                largestContour = cnts[i]
            

    Bin_WithContour = cv2.drawContours(cv2.cvtColor(Bin,cv2.COLOR_GRAY2RGB), cnts, index, (0,255,0), 1)
    t = 2
    x, y, a, b = cv2.boundingRect(largestContour)
    isConvex = cv2.isContourConvex(largestContour)
    cv2.rectangle(Bin_WithContour, (x, y), (x + a - t, y + b - t), (0,0,255), t)
    cv2.imshow("Binary with contour", Bin_WithContour)
    
    
    cv2.rectangle(gray, (340,50),(370,80),(0,255,0),2)
    cv2.imshow("depthframe",gray)
    masked_image = cv2.bitwise_and(gray, Bin)
    cv2.imshow("masked image", masked_image)

    t1_3 = cv2.getTrackbarPos("canny_thr1_3", "sliders")
    t2_3 = cv2.getTrackbarPos("canny_thr2_3", "sliders")
    canny_binary = cv2.Canny(Bin, t1_3, t2_3)

    cv2.imshow("canny on otsu",canny_binary)
    
    depth_list = list()

    key = cv2.waitKey(1)
    if key == 27:
        avg_th1 = find_depth2(largestContour, originaldepth, ground_depth, 15)
        avg_th2, med_th, avg_th3, avg_th4 = find_depth(largestContour, ground_depth, originaldepth, Bin, 13, 5, 15)
        print("Area: " + "%.3f" % (cv2.contourArea(largestContour)*(measurement_scale**2)) + " mm2\tPerimeter: " + "%.3f" % (cv2.arcLength(largestContour, True)*measurement_scale) + " mm\tThickness: " + "%.3f" % avg_th1 + " | " + "%.3f" % avg_th2 + " | " + "%.3f" % avg_th3 + " | " + "%.3f" % avg_th4)
        cv2.destroyAllWindows()
        break

