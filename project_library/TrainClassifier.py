from LoadData import *
from ExtractFeatures import *
#import matplotlib.image as mpimg
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import time
import pickle



if __name__ == '__main__':
    print("Read image log names ....")
    # vehicles
    path1 = '../TrainTestData/Initial_GKU/vehicles/GTI_Far/image*.png'
    path2 = '../TrainTestData/Initial_GKU/vehicles/GTI_Left/image*.png'
    path3 = '../TrainTestData/Initial_GKU/vehicles/GTI_MiddleClose/image*.png'
    path4 = '../TrainTestData/Initial_GKU/vehicles/GTI_Right/image*.png'
    path5 = '../TrainTestData/Initial_GKU/vehicles/KITTI_extracted/*.png'
    vehicle_paths = [path1,path2,path3,path4,path5]
    # non vehicles
    path6 = '../TrainTestData/Initial_GKU/non_vehicles/Extras/extra*.png'
    path7 = '../TrainTestData/Initial_GKU/non_vehicles/GTI/image*.png'
    nonvehicle_paths = [path6,path7]
    # build training and testing data set
    train_names,train_labels,test_names,test_labels = BuildTrainTest(vehicle_paths,nonvehicle_paths)
    print("Number of training: ", len(train_names))
    print("Number of testing:  ", len(test_names))
    
    # extract train/test features #HLS,HSV,LUV,RGB,YCrCb
    feature_color = {'hog':'YCrCb','spatial':'RGB','hist':'YCrCb'}     
    #feature_color = {'hog':'YCrCb','spatial':'YCrCb','hist':'YCrCb'} 
    #feature_color = {'hog':'YCrCb','spatial':'HSV','hist':'HSV'}
    #feature_color = {'hog':'YCrCb','spatial':'RGB','hist':'RGB'}
    #feature_color = {'hog': 'YCrCb','spatial': 'RGB','hist': 'HSV'}
    orient = 9                  # HOG orientations
    pix_per_cell = 8            # HOG pixels per cell
    cell_per_block = 2          # HOG cells per block
    hog_channel = 'ALL'         # HOG channel, can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32)     # Spatial binning dimensions
    hist_bins = 32              # Number of histogram bins
    spatial_feat = True         # Spatial features on or off
    hist_feat = True            # Histogram features on or off
    hog_feat = True             # HOG features on or off
    
    print("Extract features ....")
    # Check the training time 
    t=time.time()
    
    train_features = extract_features(train_names, color_space=feature_color, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    
    test_features = extract_features(test_names, color_space=feature_color, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    
    # Feature scalling
    X = np.vstack((train_features, test_features)).astype(np.float64) 
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Get scaled training/testing dataset
    X_train = scaled_X[:len(train_features)]
    X_test  = scaled_X[len(train_features):]
    # Feature labels
    y_train = train_labels
    y_test  = test_labels
    
    '''
    plt.subplot(2,2,1)
    plt.plot(train_features[0])
    plt.subplot(2,2,2)
    plt.plot(X_train[0])
    plt.subplot(2,2,3)
    plt.plot(test_features[0])
    plt.subplot(2,2,4)
    plt.plot(X_test[0])
    plt.show()
    print("Number of training: ", len(X_train))
    print("Number of testing:  ", len(X_test))
    '''
    
    print("Fitting classifier ....")
    # Use a linear SVC
    '''
    svc = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge',
              max_iter=1000, multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=0)
    '''
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    
    # Save calibration data for later use
    used_features = [spatial_feat, hist_feat, hog_feat]
    svc_classifier = {'svc':svc, 'scaler':X_scaler, 'orient':orient,
                      'pix_per_cell':pix_per_cell,'cell_per_block':cell_per_block,
                      'spatial_size':spatial_size,'hist_bins':hist_bins,'feature_color':feature_color,
                      'hog_channel':hog_channel,'spatial_feat':spatial_feat,'hist_feat':hist_feat,'hog_feat':hog_feat}
    
    with open('svc_classifier.pkl','wb') as f:
        pickle.dump(svc_classifier,f)
        print('Data save to local disk....')

    
    
    
    

    
    






