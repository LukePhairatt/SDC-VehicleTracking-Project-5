from LoadData import *
from ExtractFeatures import *
#import matplotlib.image as mpimg
from sklearn.svm import LinearSVC
from sklearn import svm, grid_search
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import time
import pickle


if __name__ == '__main__':
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
    
    # extract train/test features
    hog_channel = "ALL"         # HOG channel, can be 0, 1, 2, or "ALL"
    orient = 8                  # HOG orientations
    pix_per_cell = 8            # HOG pixels per cell
    cell_per_block = 2          # HOG cells per block
    spatial_size = (32, 32)     # Spatial binning dimensions
    hist_bins = 32              # Number of histogram bins
    spatial_feat = True         # Spatial features on or off
    hist_feat = True            # Histogram features on or off
    hog_feat = True             # HOG features on or off
    
    # feature color spaces to be tested
    hog_f = ['HSV','LUV','RGB','YCrCb']
    spatial_f = ['HSV','LUV','RGB','YCrCb']
    hist_f = ['HSV','LUV','RGB','YCrCb']
    
    # run it n time and do the average of the top 5 results
    total = np.zeros(64)
    for run in range(5):
        # shuffle data 
        train_names,train_labels = shuffle(train_names,train_labels)
        test_names,test_labels = shuffle(test_names,test_labels)
        # results history
        score_list = []
        feature_list = []
        for i in range(len(hog_f)):
            for j in range(len(spatial_f)):
                for k in range(len(hist_f)):
                    feature_color = {'hog':hog_f[i],'spatial':spatial_f[j],'hist':hist_f[k]}
                    # Check the training time    
                    train_features = extract_features(train_names[0:400], color_space=feature_color, 
                                        spatial_size=spatial_size, hist_bins=hist_bins, 
                                        orient=orient, pix_per_cell=pix_per_cell, 
                                        cell_per_block=cell_per_block, 
                                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                        hist_feat=hist_feat, hog_feat=hog_feat)
                    
                    test_features = extract_features(test_names[0:400], color_space=feature_color, 
                                        spatial_size=spatial_size, hist_bins=hist_bins, 
                                        orient=orient, pix_per_cell=pix_per_cell, 
                                        cell_per_block=cell_per_block, 
                                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                        hist_feat=hist_feat, hog_feat=hog_feat)
                    
                    # Feature scalling
                    X_scaler_train = StandardScaler().fit(train_features)
                    X_train = X_scaler_train.transform(train_features)
                    X_scaler_test = StandardScaler().fit(test_features)
                    X_test = X_scaler_test.transform(test_features)
                    # Feature labels
                    y_train = train_labels[0:400]
                    y_test  = test_labels[0:400]
                    
                    # Use a linear SVC 
                    svc = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge',
                              max_iter=1000, multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=0)
                    
                    
                    svc.fit(X_train, y_train)
                    # Check the score of the SVC
                    acc = round(svc.score(X_test, y_test), 5)
                    print('Accuracy = ', acc, 'type = ', feature_color)
                    
                    score_list.append(acc)
                    feature_list.append(feature_color)
        
        total = total + np.array(score_list)
          
    # acc average over 5 runs
    avg = total/5.0
    features = np.array(feature_list)
    # find the best n result from the sorted array reverse(min to max)
    indx = (-avg).argsort()[:5]
    print("best 5 config", features[indx], "score = ", avg[indx])
    
    #best 5 configs 
    #[{'hog': 'LUV', 'hist': 'RGB', 'spatial': 'HSV'}
    #{'hog': 'YCrCb', 'hist': 'RGB', 'spatial': 'HSV'}
    #{'hog': 'LUV', 'hist': 'HSV', 'spatial': 'HSV'}
    #{'hog': 'YCrCb', 'hist': 'HSV', 'spatial': 'HSV'}
    #{'hog': 'LUV', 'hist': 'YCrCb', 'spatial': 'HSV'}] 
    #score =  [ 0.9865  0.9855  0.985   0.985   0.9845]
    
    
    #
    # Search svm configuration on the best configuration
    #
    feature_color = {'hog':'YCrCb','spatial':'RGB','hist':'YCrCb'}
    train_features = extract_features(train_names[0:800], color_space=feature_color, 
                                        spatial_size=spatial_size, hist_bins=hist_bins, 
                                        orient=orient, pix_per_cell=pix_per_cell, 
                                        cell_per_block=cell_per_block, 
                                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                        hist_feat=hist_feat, hog_feat=hog_feat)
                    
    test_features = extract_features(test_names[0:800], color_space=feature_color, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    
    # Feature scalling
    X_scaler_train = StandardScaler().fit(train_features)
    X_train = X_scaler_train.transform(train_features)
    X_scaler_test = StandardScaler().fit(test_features)
    X_test = X_scaler_test.transform(test_features)
    # Feature labels
    y_train = train_labels[0:800]
    y_test  = test_labels[0:800]
    
    parameters = {'kernel':('linear'), 'C':[0.1, 1, 5, 10]}
    svr = svm.SVC()
    clf = grid_search.GridSearchCV(svr, parameters)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    #{'kernel': 'linear', 'C': 0.1}
    
    

    
  
    


    
    
    
    

    
    






