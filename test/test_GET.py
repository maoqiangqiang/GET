import torch 
import numpy as np
import pandas as pd
import time
import json

import sys
sys.path.append('./src/')

from dataset import loadDataset 
from treeFunc import readTreePath, objv_cost
from warmStart import CARTRegWarmStart
from GET_IterAlp import  multiStartTreeOptbyGRAD_withC


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
    epochNum = int(sys.argv[6])                     #  e.g. 1000; large than 21 epoch
    deviceArg =  str(sys.argv[7])                   #  "cuda" or "cpu"
    device = torch.device(deviceArg)
    startNum = int(sys.argv[8])                     #  e.g. 1, 2, 3, 4, 5...



    ##  data
    datasetPath = "./data/"
    # all datasets (all n>1000)
    DatasetsNames = ["airfoil-self-noise", "space-ga", "abalone", "gas-turbine-co-emission-2015", "gas-turbine-nox-emission-2015",  "puma8NH",  "cpu-act", "cpu-small", "kin8nm", "delta-elevators", "combined-cycle-power-plant", "electrical-grid-stability", "condition-based-maintenance_compressor", "condition-based-maintenance_turbine", "ailerons", "elevators", "houses", "house-8L", "house-16H", "friedman-artificial", "protein-tertiary-structure",  "nasa-phm2008-1",  "power-consumption-tetouan-zone1", "power-consumption-tetouan-zone2", "power-consumption-tetouan-zone3"]


    # read the treePath from the HDF5 file
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

            # cart warm start
            aInit, bInit, cInit = CARTRegWarmStart(X, Y, treeDepth, device)
            cartWarmStart_dict = {"a": aInit, "b": bInit, "c": cInit}
            treeCART = {"a": torch.tensor(aInit, device=device), "b": torch.tensor(-bInit, device=device), "c": torch.tensor(cInit, device=device)}
            objvMSECART, r2CART = objv_cost(X, Y, treeDepth, treeCART)
            objvMSECART_test, r2CART_test = objv_cost(X_test, Y_test, treeDepth, treeCART)
            print("objvMSECART_train: {};   r2CART_train: {}".format(objvMSECART, r2CART))
            print("objvMSECART_test: {};    r2CART_test: {}".format(objvMSECART_test, r2CART_test))



            warmStart = [cartWarmStart_dict]
            objv_GETCur, treeGETCur = multiStartTreeOptbyGRAD_withC(X, Y, treeDepth, indices_flags_dict, epochNum, device, warmStart, startNum)


            objvMseTrainGET, r2TrainGET = objv_cost(X, Y, treeDepth, treeGETCur)
            objvMseTestGET, r2TestGET = objv_cost(X_test, Y_test, treeDepth, treeGETCur)



            elapsedTime = time.perf_counter() - startTime




            ## final results
            print("\nFinal Results...")
            print("\nobjvMseTrain: {};    objvMseTest: {}".format(objvMseTrainGET, objvMseTestGET))
            print("r2Train: {};   r2Test: {}".format(r2TrainGET, r2TestGET))
            print("\nelapsedTime: {}\n".format(elapsedTime))


