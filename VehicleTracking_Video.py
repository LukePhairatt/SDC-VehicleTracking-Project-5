import pickle
from moviepy.editor import VideoFileClip
from project_library.HogHeatMap import *
from project_library.CommonLib import *
import glob
import matplotlib.pyplot as plt




class SVM_VehicleTracking():
    def __init__(self,svc,X_scaler,orient,pix_per_cell,cell_per_block,
                 spatial_size,hist_bins,feature_color,hog_channel,
                 spatial_feat,hist_feat,hog_feat,ystart,ystop,scales):
        self.svc            = svc
        self.X_scaler       = X_scaler
        self.orient         = orient
        self.pix_per_cell   = pix_per_cell
        self.cell_per_block = cell_per_block
        self.spatial_size   = spatial_size
        self.hist_bins      = hist_bins
        self.feature_color  = feature_color
        self.hog_channel    = hog_channel
        self.spatial_feat   = spatial_feat
        self.hist_feat      = hist_feat
        self.hog_feat       = hog_feat
        self.enabled_features  = [spatial_feat, hist_feat, hog_feat]
        self.ystart         = ystart
        self.ystop          = ystop
        self.scales         = scales
        self.Global_heatmap             = []
        self.Global_heatmap_filtered    = []
        self.Current_heatmap            = []
        self.min_threshold = -1     # Globalmap penalty on the empty cells (outside bounding box)
        self.max_threshold = 25     # Globalmap pixels maximum value
        self.object_value  = 2      # Globalmap reward on the empty cells (outside bounding box)
        self.GB_threshold  = 20     # Globalmap thresholding value to remove false predictions
        self.frame_i       = 0    
        
        
        
    def TrackingPipeline(self, img):
        self.frame_i += 1
        # STEP 1: SVM Classification
        # Hog subsampling method to detect cars
        out_img, box_list = find_cars(img, self.ystart, self.ystop, self.scales, self.svc, self.X_scaler,
                                      self.orient,self.pix_per_cell, self.cell_per_block, self.spatial_size,
                                      self.hist_bins,self.feature_color, self.hog_channel, self.enabled_features)
        # STEP 2: Current heatmap 
        # Record Current frame found pixels on the current heat map for visualisation
        Current_heat = np.zeros_like(img[:,:,0]).astype(np.float)
        # Add heat to each box in box list
        Current_heat = add_heat(Current_heat,box_list)
        # Apply threshold to help remove false positives
        Current_heat = apply_threshold(Current_heat,1)
        # Visualize the heatmap when displaying    
        self.Current_heatmap = np.clip(Current_heat, 0, 255)
        
        # STEP 3: Global heatmap update
        # Record result in the global heat map for every run
        if len(self.Global_heatmap) == 0:
            self.Global_heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
            self.Global_heatmap_filtered =  np.copy(self.Global_heatmap)
        
        # Update global map with the found or not found pixels
        # Note: we need to de this on every frame, eventhough nothing found
        #       so we reduce the heat intensity to remove false prediction early on
        self.Global_heatmap = update_globalmap(self.Global_heatmap, box_list)
       
        # STEP 4: Global heatmap filtering false positive
        # Filtering cars from the global map : anything below the threshold will be remove
        # e.g. false positive will be remove
        self.Global_heatmap_filtered = np.copy(self.Global_heatmap)
        self.Global_heatmap_filtered = apply_threshold(self.Global_heatmap_filtered, self.GB_threshold)
        
        # STEP 5: Output display
        # Find final boxes from heatmap using label function
        # labels[0] is label id  in the shape of heat matrix
        # labels[1] is the number of labels (start from 1 if found)
        labels = label(self.Global_heatmap_filtered)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        
        # STEP 6: Add global heatmap onto the image
        x_offset = 1045
        y_offset = 20
        scaling = 6
        draw_img = AddPictures(draw_img, np.copy(self.Global_heatmap).astype(np.uint8), x_offset, y_offset, scaling)
        return draw_img
        
        


if __name__ == '__main__':
    # Read training data configuration
    dist_pickle = pickle.load( open( "./project_library/svc_classifier.pkl", "rb" ) )
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    feature_color = dist_pickle["feature_color"]
    hog_channel = dist_pickle['hog_channel']
    spatial_feat = dist_pickle['spatial_feat']
    hist_feat = dist_pickle['hist_feat']
    hog_feat = dist_pickle['hog_feat']
    enabled_features = [spatial_feat, hist_feat, hog_feat]
    # Raw search area / scales  
    ystart = 360
    ystop  = 640
    scales = [1.3, 1.5, 2.0]      
    print(orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, feature_color, enabled_features, hog_channel)
   
    # SVM Detection
    VTracking = SVM_VehicleTracking(svc,X_scaler,orient,pix_per_cell,cell_per_block,
                 spatial_size,hist_bins,feature_color,hog_channel,
                 spatial_feat,hist_feat,hog_feat,ystart,ystop,scales)
    
    # Video processing
    drive_output = 'project_video_out.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    drive_clip = clip1.fl_image(VTracking.TrackingPipeline) #NOTE: this function expects jpg color images!!
    drive_clip.write_videofile(drive_output, audio=False)
       
        
    
    
    