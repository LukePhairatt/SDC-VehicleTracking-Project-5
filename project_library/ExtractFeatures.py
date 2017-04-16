'''
    Extract spatial and hog feature functions
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import glob
import numpy as np
import cv2
from skimage.feature import hog


# Convert RGB to specific color type
def convert_color(img, color_space):    
    if color_space == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = img
    return feature_image


def convertTo255(img):
    img_cvt = np.array([x*255 for x in img]).astype(np.uint8)
    return img_cvt

def ContrastEnhanceRGB(img):
    # Convert to YUV
    img_yuv = convert_color(img, 'YUV')
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert  back to RGB format
    img_en = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    img_en = cv2.blur(img_en,(5,5))
    return img_en
    
    
    
# Define a function to return HOG features and visualization
# Note img is 1 channel only: could be a type of RGB, HLS, YCrCb....
def get_hog_features(img, orient, pix_per_cell, cell_per_block, hog_vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if hog_vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=hog_vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=hog_vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32), bin_vis=False):
    # image resize
    img_resize = cv2.resize(img, size)
    # Create the feature vector
    features = img_resize.ravel() 
    if bin_vis:
        return features, img_resize
    else:
        # Return the feature vector
        return features

# Define a function to compute color histogram features 
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space, spatial_size=(32, 32), hist_bins=32, orient=9, 
                     pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True,
                     hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        # Using matplotlib
        #   jpg will be 0-255
        #   png will be 0-1.0
        image = mpimg.imread(file)
        img = np.copy(image)
        
        # I am using 0-255 scale through out
        # convert png 0-1.0 scale to 0-255
        img = convertTo255(img)
        
        # enhance contrast
        #img = ContrastEnhanceRGB(img)
    
        if spatial_feat == True:
            # convert to specific color type for spatial features
            feature_image = convert_color(img, color_space['spatial']) 
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # convert to specific color type for histogram features
            feature_image = convert_color(img, color_space['hist'])
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # convert to specific color type for hog features
            feature_image = convert_color(img, color_space['hog'])
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        hog_vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, hog_vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        
        # combine all feature spaces
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
    
def MultiplePlots(grid, images, titles):
    nrow = grid[0]
    ncol = grid[1]
    plt.figure(figsize=(ncol+1, nrow+1)) 
    gs = gridspec.GridSpec(nrow, ncol,
             wspace=0.2, hspace=0.2, 
             top=1.-0.2/(nrow+1), bottom=0.2/(nrow+1), 
             left=0.2/(ncol+1), right=1-0.2/(ncol+1))

    for i in range(nrow):
        for j in range(ncol):
            ax = plt.subplot(gs[i,j])
            img = images[i][j]
            title = titles[i][j]
            ax.imshow(img)
            ax.set_title(title)
            xticks = np.arange(0, img.shape[1], 10)
            yticks = np.arange(0, img.shape[0], 10)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels(xticks)
            ax.set_yticklabels(yticks)
    plt.show()
    
    
def test_bin_spatial(img, color_space, spatial_size):
    # convert to specific color type for spatial features
    feature_image = convert_color(img, color_space['spatial'])
    # extract features
    spatial_features_ch0, spatial_image_ch0 = bin_spatial(feature_image[:,:,0], size=spatial_size, bin_vis=True)
    spatial_features_ch1, spatial_image_ch1 = bin_spatial(feature_image[:,:,1], size=spatial_size, bin_vis=True)
    spatial_features_ch2, spatial_image_ch2 = bin_spatial(feature_image[:,:,2], size=spatial_size, bin_vis=True)
    # Grid plot
    grid = [3,2]
    images = [[feature_image[:,:,0],spatial_image_ch0],
              [feature_image[:,:,1],spatial_image_ch1],
              [feature_image[:,:,2],spatial_image_ch2]]
    titles = [[color_space['spatial'] + ' CH-0',color_space['spatial'] + ' CH-0 FEATURE'],
              [color_space['spatial'] + ' CH-1',color_space['spatial'] + ' CH-1 FEATURE'],
              [color_space['spatial'] + ' CH-2',color_space['spatial'] + ' CH-2 FEATURE']]
    MultiplePlots(grid, images, titles)
    
def test_hog_feature(img, color_space, orient, pix_per_cell, cell_per_block):
    # convert to specific color type for hog features
    feature_image = convert_color(img, color_space['hog'])
    # Call get_hog_features() with vis=False, feature_vec=True
    hog_features_ch0, hog_image_ch0 = get_hog_features(feature_image[:,:,0], orient, pix_per_cell, cell_per_block, hog_vis=True, feature_vec=True)
    hog_features_ch1, hog_image_ch1 = get_hog_features(feature_image[:,:,1], orient, pix_per_cell, cell_per_block, hog_vis=True, feature_vec=True)
    hog_features_ch2, hog_image_ch2 = get_hog_features(feature_image[:,:,2], orient, pix_per_cell, cell_per_block, hog_vis=True, feature_vec=True)
    
    # Grid Plot
    grid = [3,2]
    images = [[feature_image[:,:,0],hog_image_ch0],
              [feature_image[:,:,1],hog_image_ch1],
              [feature_image[:,:,2],hog_image_ch2]]
    
    titles = [[color_space['hog'] + ' CH-0',color_space['hog'] + ' CH-0 HOG'],
              [color_space['hog'] + ' CH-1',color_space['hog'] + ' CH-1 HOG'],
              [color_space['hog'] + ' CH-2',color_space['hog'] + ' CH-2 HOG']]
    
    MultiplePlots(grid, images, titles)

    
if __name__ == '__main__':
    feature_color = {'hog':'YCrCb','spatial':'HSV','hist':'YCrCb'} #HLS,HSV,LUV,RGB,YCrCb
    orient = 8                  # HOG orientations
    pix_per_cell = 8            # HOG pixels per cell
    cell_per_block = 2          # HOG cells per block
    hog_channel = "ALL"         # HOG channel, can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32)     # Spatial binning dimensions
    hist_bins = 32              # Number of histogram bins
    spatial_feat = True         # Spatial features on or off
    hist_feat = True            # Histogram features on or off
    hog_feat = True             # HOG features on or off
    
    # read test images
    vehicle = glob.glob('../TrainTestData/Initial_GKU/vehicles/GTI_MiddleClose/image*.png')
    notvehicle = glob.glob('../TrainTestData/Initial_GKU/non_vehicles/Extras/extra*.png')
    # randomly pick one img
    vehicle_indx = 100#np.random.randint(1,len(vehicle))
    notvehicle_indx = 100#np.random.randint(1,len(notvehicle))
    vehicle_img = mpimg.imread(vehicle[vehicle_indx])
    notvehicle_img = mpimg.imread(notvehicle[notvehicle_indx])
    # I am using 255 scale through out the project
    vehicle_img = convertTo255(vehicle_img)
    notvehicle_img = convertTo255(notvehicle_img)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(vehicle_img)
    plt.title('Car')
    plt.subplot(1,2,2)
    plt.imshow(notvehicle_img)
    plt.title('Not car')
    plt.show()
    

    # show result: car
    test_hog_feature(vehicle_img, feature_color, orient, pix_per_cell, cell_per_block)
    test_bin_spatial(vehicle_img, feature_color, spatial_size)
    # show result: not car
    test_hog_feature(notvehicle_img, feature_color, orient, pix_per_cell, cell_per_block)
    test_bin_spatial(notvehicle_img, feature_color, spatial_size)
    
   
   
    
    
    
    