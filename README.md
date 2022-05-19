# MTSPL-CV

A Computer Vision approach to find thickness of metal scrap pieces.
The depth data is extracted from RGBD frames of an Intel Realsense Camera (D435i).

## Usage:
- Run oneframe.py as "python -u path_to_oneframe.py -i path_to_bag_file"
- once run, some images will be shown which depict some stages in processing and calculation.
- press 'esc' to exit the runtime and get results for that particular object.
- Results include:
  - Mean Ground Depth used.
  - Detected Area of the object in mm^2.
  - Perimeter of the detected area in mm.
  - Thickness in mm by the 4 approaches.

![image](https://user-images.githubusercontent.com/56913610/169297015-3ad4fcb5-3986-45dd-b43c-0a70b887c340.png)                       
_results of dc1.bag_


## Thickness Calculation
#### Approach 1: |Mean ground Depth - Mean Depth at Edge|
Considers on the complete identified perimeter. This avoids any surface curvature in the given object. The difference of the Average depth of the object and the average background depth is taken as the thickness of the detected object.

![image](https://user-images.githubusercontent.com/56913610/169398991-7fc58c00-5054-4fbf-8888-a3868199fd49.png)                                                                  
_image depicting pixels under consideration for determination of thickness_

#### Approach 2: maximum interclass variance technique for image segmentation. threshold = 0.15 for best results.
Regions near the boundary/interface of background and object are given importance. A certain x % interclass variance is used in finding thickness of the object. This takes care of any abnormal curvature of the piece as that will increase the rate of change of depth, but over multiple class intervals, which won't affect the highest/most weighted-interclass variance.

![image](https://user-images.githubusercontent.com/56913610/169399651-7339c6fd-a487-4ba1-aee0-f2673249d778.png)                                                                                 
_image depicting pixel depth comparisons under consideration for the local edge pixel_

#### Approach 3: |Mean Ground Depth - Mean depth of Inner Edge Points|
Similar to approach 1, but only white pixels near boundary/interface regions are considered. This removes any false thickness due to uplifted scrap pieces due to intrinsic curvature.

![image](https://user-images.githubusercontent.com/56913610/169400297-75a614d4-8e80-42c7-bb87-39f724ddf3c8.png)                                                                    
_pixels on the inner side of white region are considered for comparison and thickness calculation_

#### Approach 4: |Mean Ground Depth - Mean Depth of Outer Edge Points|
Similar to approach 1, but only black pixels near boundary/interface regions are considered. Equivalent to approach 3, with the added advantage to account for metal scrap edges that are flush to the background.

![image](https://user-images.githubusercontent.com/56913610/169400892-6138eed5-0569-45dd-8611-eda9443e321a.png)                                                                 
_pixels on the outer side of white region are considered for comparison and thickness calculation_



![image](https://user-images.githubusercontent.com/56913610/169299897-5ab5d439-d1c8-4f67-b7d6-de2418686d18.png)
_thickness calculation method on a depth histogram_


## Width Calculation
Mean Width = length of convex hull / &pi;





## More Files for testing
The directory has 10 sample '.bag' files for testing. Other '.bag' can be found at: https://drive.google.com/drive/folders/1LKwdeZ-oOIrMPxQqLpQ3Fcminoj9NOtf?usp=sharing
