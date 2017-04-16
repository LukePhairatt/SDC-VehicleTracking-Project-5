
# **Vehicle Detection Project-5 SDC**
![alt text][image0]

The goals / steps of this project are the following:
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* A color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize these features and randomise and balance a selection for training and testing
* Train Support Vector Machine for the car and non car classification
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
* [**Rubric Points**](https://review.udacity.com/#!/rubrics/513/view)

[//]: # (Image References)
[image0]: ./output_images/intro.png "project"
[image1]: ./output_images/features/data_in.png "Cars-Notcars"
[image2]: ./output_images/features/car_hog.png "Car-HOG" 
[image3]: ./output_images/features/car_spatial.png "Car-Spatial"
[image4]: ./output_images/features/notcar_hog.png "Notcar-HOG"
[image5]: ./output_images/features/notcar_spatial.png "Notcar-Spatial"
[image6]: ./output_images/ParametersTuning/hog_0_spatial_16.png "HOG0"
[image7]: ./output_images/ParametersTuning/hog_1_spatial_16.png "HOG1"
[image8]: ./output_images/ParametersTuning/hog_2_spatial_16.png "HOG2"
[image9]: ./output_images/ParametersTuning/hog_all_spatial_16.png "HOG_ALL_LOW"
[image10]: ./output_images/ParametersTuning/hog_all_spatial_32.png "HOG_ALL_GOOD"
[image11]: ./output_images/tracking/all.png "SampleResults"
[image12]: ./output_images/windowsearch/HogSubsampling.png "HOG_Search"
[video1]: ./project_video_out.mp4

 
### Training Data visualisation: Car and Noncar image ###
![alt text][image1]
I extracted the training/testing data from the image files. I balanced the cars and non cars images on both training and testing data set(GTI* files). In addition, the time- series car images are sorted in the time series order using the image tag names. I then split the first 80% for training and bottom 20% for testing. So the training data is different from the testing data to avoid over-fitting. The code for this is in BuildImageNames() function in project_library/LoadData.py.


### Histogram of Oriented Gradients (HOG)
To get some idea on what the color spaces (e.g. YCrCb, LUV, HSV and etc) that I will use for this project. I tried the combination of these color spaces in the SVM training process to gauge what the color spaces and combination give the good performance (accuracy). The codes for this color channel search are in project_library/GridSearchTrainClassifier.py.

I was settle with HOG:YCrCb, Spatial:RGB, Histogram:YCrCb

Having found the color space combination, I then next worked on the paramters for HOG, Spatial size and Histogram bin.
The results from few tries are in this [**folder**](./output_images/ParametersTuning/)

![alt text][image10]
I then settled with the paramaters with this resut above.


### Linear SVM 
I only use SVM for the classification. The model was trained on the GTI data set
 
Data summary: Balanced data set
training car:  7032, training not car:  7174
testing car:  1760, testing not car:  1794
Number of training data:  14206
Number of testing data:   3554

The accuracy was about 0.98-0.99 on the defalut SVM parameters

### Sliding Window Search: Hog sub-sampling
I used the settings as follows:
cells_per_pixel = 8
cell_per_block  = 2 
cells_per_step = 2 (100*6/8 = 75% overlap = 0.75*64 = 48 pixels or 16 pixels apart)
Search area: ystart = 360, ystop  = 640
Scales = [1.3, 1.5, 2.0] 

These scalings are estimated from the sizes of the vehicles that might appear on the images and we want to track them. 
To get some idea on the sizes and scaling by the trained image of 64x64 pixels 

car size big:    width = 200 pixels, height = 100 pixels  →  scale x = 3.1, scale y = 1.6
car size medium: width = 140 pixels, height =   65 pixels  →  scale x = 2.2, scale y = 1.0
car size small:  width = 100 pixels, height =   50 pixels  →  scale x = 1.6, scale y = 0.8

The trained imaged size (cars) can be varied. So the scales was set according to these ranges. 
The above scales of 1.3, 1.5, 2.0 provided the good result in the detection for this implemented solution.

In addition, the window overlap was set to  75% to enhance a number of detections on the same vehicle. So we have a good portion of positive detections to build the more robust heat map and filtering. The result on the test images are display belows.

![alt text][image12]


### Video Implementation
![link to my video result][video1]


### False prediction filtering
On filtering the false positive predictions, I used the similar approach as advised in the lesson. I slightly modified this technique to the global heat map update approach (ehicleTracking_Video.py and update_globalmap() function implemented in HogHeatMap.py) to keep track of the car positions overtime. In this approach, I recorded the positions of positive detections in each frame of the video by incrementing the pixel value by 1 and decreasing other area (negative detections) by 1 as well. So the heat value and position is building and decreasing overtime. The limit of the heat values is controlled by the min and max threshold parameter.  This limit will control the responsiveness in the vehicle detection and tracking (VehicleTracking_Video.py).
  
After the global map is updated on the current detection, it is then thresholded to filter out any false detection and to identify the vehicle positions (VehicleTracking_Video.py, apply_threshold() in HogHeatMap.py). I then used ‘scipy.ndimage.measurements.label()’ to identify individual blobs in the heatmap.  I then assumed each blob corresponded to the vehicle.  I constructed bounding boxes to cover the area of each blob detected (draw_labeled_bboxes() in HogHeatMap.py).

Here's an example result showing the global and filtered heatmap from a series of frames of video, the result of ‘scipy.ndimage.measurements.label()’ and the bounding boxes then update/overlaid on every frame.

![alt text][image11]


### Discussion

In this project, I used the support vector machine (SVM) classifier for detecting the vehicles in the image frame. The extracted features using Hog, Spatial color and Histogram for training the classifier are able to produce the good predictions. However, there are some places that the classifiers producing the false positive results due to changing conditions. Even though we can filter this out using the recorded heat map as implemented in this project or something similar. It still does not guarantee that these false detections would not slip through.

This implemented solution might fail in the places where the image contrast is low such as driving through overcasting shadow or the brightness is changed all the sudden. The potential solution to this problem is to train the classifier with these types of images.

For the future work, I believe a deep learning method such as Neural Network might produce a better result in recognising the cars from non-cars. In addition, we could use more than one classifier to weigh the prediction so we could filter out the low probability for the final output.

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

