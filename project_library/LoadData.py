import numpy as np
import glob
from sklearn.utils import shuffle


def BuildImageNames(Paths, test_size=0.2):
    train_names = []
    test_names   = []
    for path in Paths:
        # load file name and sort them in order before spliting data
        # Note: data is in a time order so sorting and spliting them
        #       will make sure the train and test data are quite different
        all_names = np.sort(glob.glob(path))
                       
        if(len(all_names)==0):
            print("File not found....")
        # split train & test data
        split = np.int((1.0-test_size) * len(all_names))
        train_names.append(all_names[0:split])
        test_names.append(all_names[split:])
        
    # merge list
    train_names = np.concatenate(train_names)
    test_names = np.concatenate(test_names)
    # shuffle list
    train_names = shuffle(train_names)
    test_names  = shuffle(test_names)
    return train_names, test_names

def BuildTrainTest(path_vehicles, path_notvehicles):
    # vehicles: label 1
    train_vehicle_names, test_vehicle_names   = BuildImageNames(path_vehicles)
    train_vehicle_labels, test_vehicle_labels = np.ones(len(train_vehicle_names)), np.ones(len(test_vehicle_names))
    # non venicles: label 0
    train_notvehicle_names, test_notvehicle_names   = BuildImageNames(path_notvehicles)
    train_notvehicle_labels, test_notvehicle_labels = np.zeros(len(train_notvehicle_names)), np.zeros(len(test_notvehicle_names))
    # merge training/testing data and labels and shuffle them together
    train_X  = np.hstack((train_vehicle_names, train_notvehicle_names))
    train_y = np.hstack((train_vehicle_labels, train_notvehicle_labels))
    train_names, train_labels = shuffle(train_X, train_y)
    test_X = np.hstack((test_vehicle_names, test_notvehicle_names))
    test_y = np.hstack((test_vehicle_labels, test_notvehicle_labels))
    test_names, test_labels   = shuffle(test_X, test_y)
    
    print("training car: ", len(np.argwhere(train_y > 0)), "training not car: ", len(np.argwhere(train_y < 1))  )
    print("testing car: ", len(np.argwhere(test_y > 0)),  "testing not car: ", len(np.argwhere(test_y < 1))  )
    return train_names,train_labels,test_names,test_labels
    
            

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
    
    
    
    
    
    