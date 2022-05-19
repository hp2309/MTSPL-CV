# MTSPL-CV

A Computer Vision approach to find thickness of metal scrap pieces.
The depth data is extracted from RGBD frames of an Intel Realsense Camera (D435i).

Usage:
- Run oneframe.py as "python -u path_to_oneframe.py -i path_to_bag_file"
- once run, some images will be shown which depict some stages in processing and calculation.
- press 'esc' to exit the runtime and get results for that particular object.
- Results include:
  - Mean Ground Depth used.
  - Detected Area of the object in mm^2.
  - Perimeter of the detected area in mm.
  - Thickness in mm by the 4 approaches.

![image](https://user-images.githubusercontent.com/56913610/169297015-3ad4fcb5-3986-45dd-b43c-0a70b887c340.png)

