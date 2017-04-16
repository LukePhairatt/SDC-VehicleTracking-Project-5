from project_library.ExtractFeatures import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label



# Mpdified from Udacity codes


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap < threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


def update_globalmap(global_map, bbox_list, min_threshold = -1, max_threshold = 30, object_value  = 2):
    # heat will be added or removed from every running frame for example 
    # +1 whatever inside the box
    # -1 whatever outside the box
    # mask template 
    template = min_threshold*np.ones_like(global_map).astype(np.float)
    # Iterate through list of bboxes
    for box in bbox_list:
        # Increment by 1
        # Add +2 for all pixels inside each bbox (becuase we start with -1)
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        template[box[0][1]:box[1][1], box[0][0]:box[1][0]] += object_value
    
    # Add mask to the global map
    global_map += template
    
    # Anything < 0 will be reset to 0
    # Otherwise empty space will be a big negative number overtime, we don't want that
    # Likewise anything > max_threshold will be limited
    global_map[global_map <= 0] = 0
    global_map[global_map >= max_threshold] = max_threshold
    return global_map

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins, color_space, hog_channel, enabled_features):
    
    # feature configuration
    used_spatial = enabled_features[0]
    used_hist    = enabled_features[1]
    used_hog     = enabled_features[2]
    
    # draw image output
    draw_img = np.copy(img)
    
    # enhance contrast
    #img = ContrastEnhanceRGB(img)
    
    # search ROI
    img_tosearch = img[ystart:ystop,:,:]
    # preprocessing feature extraction: spatial/hist/hog
    ctrans_tosearch_hog_roi  = convert_color(img_tosearch, color_space['hog'])
    ctrans_tosearch_spat_roi = convert_color(img_tosearch, color_space['spatial'])
    ctrans_tosearch_hist_roi = convert_color(img_tosearch, color_space['hist'])
    
    # found match [((x1,y1),(x2,y2))] box list
    box_list = []
    
    # search for n scales
    for scale in scales:
        # resize scalling image for a window search
        if scale != 1:
            imshape = img_tosearch.shape
            ctrans_tosearch_hog  = cv2.resize(ctrans_tosearch_hog_roi,  (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            ctrans_tosearch_spat = cv2.resize(ctrans_tosearch_spat_roi, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            ctrans_tosearch_hist = cv2.resize(ctrans_tosearch_hist_roi, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        else:
            ctrans_tosearch_hog  =  np.copy(ctrans_tosearch_hog_roi)
            ctrans_tosearch_spat =  np.copy(ctrans_tosearch_spat_roi)
            ctrans_tosearch_hist =  np.copy(ctrans_tosearch_hist_roi)
        
        # extract hog features    
        ch1 = ctrans_tosearch_hog[:,:,0]
        ch2 = ctrans_tosearch_hog[:,:,1]
        ch3 = ctrans_tosearch_hog[:,:,2]
    
        # Define a number of blocks on the INPUT image
        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1 
        #nfeat_per_block = orient*cell_per_block**2
        
        # Define a number of blocks on the TRAINED 64x64 (was the orginal size, with e.g. 8 cells (also 8 pix per cell)
        window = 64
        nblocks_per_window = (window // pix_per_cell)-1
        
        # Search resolution: How many step: overlap = (1.0-2/8)  = 0.75%
        cells_per_step = 2                                              # Instead of overlap, define how many cells to step in both x,y (min is 1) in the INPUT HOG IMAGE 
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step     # n steps in x: on the last block: end index = start index + block size, so we need to substract it out   
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step     # m steps in y: on the last block: end index = start index + block size, so we need to substract it out
        
        # Compute individual channel HOG features on the INPUT image
        # Note: we will get the hog size image data now after this! feature_vec = false
        # return hog feature of (img height/pix_per_cell)-1, (img width/pix_per_cell)-1, cell_per_block,cell_per_block, orient
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

          
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                
                # Extract HOG 'ALL' only the portion of the image (need to match match the Train 64 x 64 e.g. 7x7x2x2x8
                # Note hog shape is (img height/pix_per_cell)-1, (img width/pix_per_cell)-1, cell_per_block,cell_per_block, orient
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()            # equal to 64x64 hog TRAINED feature return e.g 7*7*2*2*8
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()            # equal to 64x64 hog TRAINED feature return e.g 7*7*2*2*8
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()            # equal to 64x64 hog TRAINED feature return e.g 7*7*2*2*8
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                
                # Spatial and Histogram data indexing
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
    
                # extract search window data (rescaling) + imitate original training size
                subimg_spat = cv2.resize(ctrans_tosearch_spat[ytop:ytop+window, xleft:xleft+window], (64,64))
                subimg_hist = cv2.resize(ctrans_tosearch_hist[ytop:ytop+window, xleft:xleft+window], (64,64))
              
                # Extract color and hist features
                spatial_features = bin_spatial(subimg_spat, size=spatial_size)
                hist_features    = color_hist(subimg_hist, nbins=hist_bins)
                
                # Scale features and make a prediction: must stack in the same order as training
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))   #original 
                test_prediction = svc.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                    box_list.append( ( (xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart) ) )
                
    return draw_img, box_list


if __name__ == '__main__':
    # Read in a pickle file with bboxes saved
    # Each item in the "all_bboxes" list will contain a 
    # list of boxes for one of the images shown above
    box_list = [((800, 400), (900, 500)), ((850, 400), (950, 500)), ((1050, 400), (1150, 500)),
                ((1100, 400), (1200, 500)), ((1150, 400), (1250, 500)), ((875, 400), (925, 450)),
                ((1075, 400), (1125, 450)), ((825, 425), (875, 475)), ((814, 400), (889, 475)),
                ((851, 400), (926, 475)), ((1073, 400), (1148, 475)), ((1147, 437), (1222, 512)),
                ((1184, 437), (1259, 512)), ((400, 400), (500, 500))]
    # Read in image similar to one shown above 
    image = mpimg.imread('bbox-example-image.jpg')
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    # labels[0] is label id  in the shape of heat matrix
    # labels[1] is the number of labels (start from 1 if found)
    labels = label(heat)
    
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    plt.show()