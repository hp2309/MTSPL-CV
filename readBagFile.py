
# First import library
from numpy.lib.type_check import imag
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0,rows):
            for y in range(0,cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0,0),None,scale,scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y],cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x],(0,0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = np.vstack(hor)

    return ver


# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
# Parse the command line arguments to an object
args = parser.parse_args()
# Safety if no parameter have been given
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()

def nothing(x):
    pass

cv2.namedWindow("sliders")

cv2.createTrackbar("canny_thr1", "sliders",113,255,nothing)
cv2.createTrackbar("canny_thr2", "sliders",123,255,nothing)
cv2.createTrackbar("mediankSize", "sliders", 14, 50, nothing)
cv2.createTrackbar("gaussiankSize", "sliders",4,20,nothing)
cv2.createTrackbar("thr", "sliders",4,255,nothing)

try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)

    # Start streaming from file
    pipeline.start(config)

    # Create opencv window to render image in
    
    # Create colorizer object
    colorizer = rs.colorizer(2.0)
    a = 0
    file1 = open("data.txt","a")
    # Streaming loop
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        # Get depth frame
        depth_frame = frames.get_depth_frame()

        rgb_frame = frames.get_color_frame()

        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        src = np.asanyarray(depth_color_frame.get_data())
        src = src[50:450, 150:850]
        # cv2.imshow("cropped", crp_src)
        mediankSize = 2*int(cv2.getTrackbarPos("mediankSize", "sliders"))+1
        gaussiankSize = 2*int(cv2.getTrackbarPos("gaussiankSize", "sliders"))+1

        kernel = np.ones((10,10), np.float32)/100
        # src_blur = cv2.filter2D(src, -1, kernel)
        src_gaussian = cv2.GaussianBlur(src, (gaussiankSize,gaussiankSize), 0)
        src_median = cv2.medianBlur(src, mediankSize)
        
        
        thr1 = cv2.getTrackbarPos("canny_thr1", "sliders")
        thr2 = cv2.getTrackbarPos("canny_thr2", "sliders")
        src_canny = cv2.Canny(src_median, thr1, thr2)
        src_canny2 = cv2.Canny(src_gaussian, thr1, thr2)


        if a == 0:
            #file1.write(str(src.tolist()))
            print(src)
            #file1.close()
            a = 1
        
        # Render image in opencv window
        # cv2.rectangle(src, (50,10),(830,460),(0,255,0),2)
        cv2.imshow("src", src)
        # cv2.imshow("blur", src_blur)
        # cv2.imshow("gaussian", src_gaussian)
        cv2.imshow("median", src_median)
        cv2.imshow("canny from median", src_canny)
        # cv2.imshow("canny from gaussian", src_canny2)
        
        #OTSU
        otsu = src_median
        hist_median = cv2.calcHist([otsu], [0], None, [256], [0,256])
        plt.plot(hist_median)
        plt.show()

        # within = []

        # for i in range(len(hist_median)):
        #     x,y = np.split(hist_median,[i])
        #     #weights
        #     x1 = np.sum(x)/(otsu.shape[0]*otsu.shape[1])
        #     y1 = np.sum(y)/(otsu.shape[0]*otsu.shape[1])
        #     #means
        #     x2 = np.sum([j*t for j,t in enumerate(x)])/np.sum(x)
        #     y2 = np.sum([j*t for j,t in enumerate(y)])/np.sum(y)
        #     #variance
        #     x3 = np.sum([(j-x2)**2*t for j,t in enumerate(x)])/np.sum(x)
        #     x3 = np.nan_to_num(x3)
        #     y3 = np.sum([(j-y2)**2*t for j,t in enumerate(y)])/np.sum(y)
        #     y3 = np.nan_to_num(y3)

        #     within.append(x1*x3 + y1*y3)
        
        # m = np.argmin(within)
        m = cv2.getTrackbarPos("thr", "sliders")
        (thresh, Bin) = cv2.threshold(otsu, m, 255, cv2.THRESH_BINARY)
        cv2.imshow("OTSU",Bin)

        # imgArray = [src,src_gaussian,src_canny]
        # imgStack = stackImages(0.5, imgArray)

        # cv2.imshow("result", (imgStack))


        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break
 
finally:
    pass