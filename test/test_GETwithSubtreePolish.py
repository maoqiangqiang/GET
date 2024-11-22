import torch 
torch.set_default_dtype(torch.float32)
import numpy as np
import pandas as pd
import time
import json 


import sys
sys.path.append('./src/')

from dataset import loadDataset
from treeFunc import readTreePath, objv_cost
from warmStart import CARTRegWarmStart
from subtreePolish import RT as subPolRT




if __name__ == "__main__":

    ################## main Code ##################
    # torch.autograd.set_detect_anomaly(True)
    
    ## Args 
    dataNumStart = int(sys.argv[1])                 # e.g. 1
    dataNumEnd = int(sys.argv[2])                   # e.g. 1
    runsNumStart = int(sys.argv[3])                 # e.g. 1
    runsNumEnd = int(sys.argv[4])                   # e.g. 1    
    
    # tree depth 
    treeDepth = int(sys.argv[5])                    #  e.g. 2 4 8 
    epochNum = int(sys.argv[6])                     #  e.g. 1000; larger than 21 epoch
    deviceArg =  str(sys.argv[7])                   #  "cuda" or "cpu"
    device = torch.device(deviceArg)
    startNum = int(sys.argv[8])                     #  e.g. 1, 2, 3, 4, 5...
    

    ##  data
    datasetPath = "./data/"
    # all datasets (all n>1000)
    DatasetsNames = ["airfoil-self-noise", "space-ga", "abalone", "gas-turbine-co-emission-2015", "gas-turbine-nox-emission-2015",  "puma8NH",  "cpu-act", "cpu-small", "kin8nm", "delta-elevators", "combined-cycle-power-plant", "electrical-grid-stability", "condition-based-maintenance_compressor", "condition-based-maintenance_turbine", "ailerons", "elevators", "houses", "house-8L", "house-16H", "friedman-artificial", "protein-tertiary-structure",  "nasa-phm2008-1",  "power-consumption-tetouan-zone1", "power-consumption-tetouan-zone2", "power-consumption-tetouan-zone3"]


    ## read the treePath from the HDF5 file
    indices_flags_dict = readTreePath(treeDepth, device)

    datasetNum = len(DatasetsNames)
    print("Starting: Total {} datasets".format(datasetNum))
    

    
    for datasetIdx in range(dataNumStart-1, dataNumEnd):
        print("############# Dataset[{}]: {} #############".format(datasetIdx+1, DatasetsNames[datasetIdx]))
        for run in range(runsNumStart, runsNumEnd+1):
            print("####### Run: {} #######".format(run))
            torch.manual_seed(run)
            np.random.seed(run)
    
            dataTrain, dataValid, dataTest = loadDataset(DatasetsNames[datasetIdx], run, datasetPath)
            
            p = dataTrain.shape[1] - 1
            X_train = torch.from_numpy(dataTrain[:, 0:p] * 1.0).float()
            Y_train = torch.from_numpy(dataTrain[:, p] * 1.0).float()
            X_valid = torch.from_numpy(dataValid[:, 0:p] * 1.0).float()
            Y_valid = torch.from_numpy(dataValid[:, p] * 1.0).float()
            X_test = torch.from_numpy(dataTest[:, 0:p] * 1.0).float()
            Y_test = torch.from_numpy(dataTest[:, p] * 1.0).float()

            X = torch.cat((X_train, X_valid), 0)
            Y = torch.cat((Y_train, Y_valid), 0)
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            # X_train = X_train.to(device, non_blocking=True)
            # Y_train = Y_train.to(device, non_blocking=True)
            # X_valid = X_valid.to(device, non_blocking=True)
            # Y_valid = Y_valid.to(device, non_blocking=True)
            X_test = X_test.to(device, non_blocking=True)
            Y_test = Y_test.to(device, non_blocking=True)



            if run == runsNumStart:
                print("dataset:{};    n_train:{};    n_valid:{};    n_test:{};    p:{}\n".format(DatasetsNames[datasetIdx], X_train.shape[0], X_valid.shape[0], X_test.shape[0], X_train.shape[1]))




            startTime = time.perf_counter()

            objv_TreeGETSubPol, TreeGETSubPol, objv_TreeGET, TreeGET, time_GETElapsed = subPolRT(X, Y, treeDepth, indices_flags_dict, epochNum, device, startNum)
            TimeGETSubPol = time.perf_counter()-startTime
            objv_MSE_train_GET, r2TrainGET = objv_cost(X, Y, treeDepth, TreeGET)
            objv_MSE_test_GET, r2TestGET = objv_cost(X_test, Y_test, treeDepth, TreeGET)

            objv_MSE_train_GETSubPol, r2TrainGETSubPol = objv_cost(X, Y, treeDepth, TreeGETSubPol)
            objv_MSE_test_GETSubPol, r2TestGETSubPol = objv_cost(X_test, Y_test, treeDepth, TreeGETSubPol)



            print("Final Results...")
            print("\nobjv_MSE_train_GET: {};    objv_MSE_test_GET: {}".format(objv_MSE_train_GET, objv_MSE_test_GET))
            print("r2TrainGET: {};    r2TestGET: {}".format(r2TrainGET, r2TestGET))
            print("\nobjv_MSE_train_GETSubPol: {};    objv_MSE_test_GETSubPol: {}".format(objv_MSE_train_GETSubPol, objv_MSE_test_GETSubPol))
            print("r2TrainGETSubPol: {};    r2TestGETSubPol: {}".format(r2TrainGETSubPol, r2TestGETSubPol))
            print("\nTimeGET: {}".format(time_GETElapsed))
            print("TimeGETSubPol: {}".format(TimeGETSubPol))


