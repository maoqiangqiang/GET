
from sklearn.tree import _tree
from sklearn import tree
import numpy as np
import torch 
import sklearn.metrics as metrics


def getNodesId(ind, Hind):
    ind -= 1                  # index starts from 0
    branchNodes = [ind]
    currentNodes = [ind]
    for _ in range(Hind-1):
        nextNodes = [2*node + j for node in currentNodes for j in [1, 2]]
        branchNodes.extend(nextNodes)
        currentNodes = nextNodes
    leftLeaf = 2*(ind+1) if Hind == 1 else 2*(nextNodes[0]+1)      # read index start from 1. eg, 32 means the 32th node, the first leaf node and index should be 31
    return branchNodes, leftLeaf


## retrieve the parameters abc of the trained tree model
def regTreeWarmStart(model, treeDepth):
    tree_ = model.tree_                  # trained model after fit
    branchNode_inputDepth = 2**(treeDepth) - 1
    # Fitted_treeDepth = model.get_depth()
    # print("Fitted_treeDepth: ", Fitted_treeDepth)
    leafNode_inputDepth = 2**(treeDepth)
    a = [0]*branchNode_inputDepth
    b = [0]*branchNode_inputDepth
    c = [0]*leafNode_inputDepth

    ab0indList = []     # save the ind when a = 0; b = 0
    def warmStartPara(node, ind):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # feature index starts from 0: 0 -> 1st feature 
            featureIdx = tree_.feature[node]
            threshold = tree_.threshold[node]
            a[ind-1] = featureIdx
            # since we use it as warmStart of our NN, we need to change the sign of the threshold
            b[ind-1] = -threshold
            
            node_l = 2 * ind
            node_r = 2 * ind + 1
            warmStartPara(tree_.children_left[node], node_l)
            warmStartPara(tree_.children_right[node], node_r)
        
        else:
            if ind <= branchNode_inputDepth:
                ## node only with one sample. it will be regrarded as a leaf node, but it might be a branch node actually. 
                currDepthForInd = int(np.log2(ind))
                diffDepthInbd = treeDepth - currDepthForInd
                ab0NodeListForEachInd, leftLeaf = getNodesId(ind, diffDepthInbd)
                ab0indList.extend(ab0NodeListForEachInd)
                c[leftLeaf-1-branchNode_inputDepth] = tree_.value[node].squeeze()
            else:
                c[ind-1-branchNode_inputDepth] = (tree_.value[node].squeeze())

    warmStartPara(0, 1)
    return a, b, c, ab0indList


def CARTRegWarmStart(X, Y, treeDepth, device):
    model = tree.DecisionTreeRegressor(max_depth=treeDepth, min_samples_leaf=1, random_state=0)
    if device == torch.device('cuda'):
        X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    else:
        X_np, Y_np = X, Y


    if X_np.shape[0] < 1:
        branchNodeNum = 2**(treeDepth) - 1
        leafNodeNum = 2**(treeDepth)
        p = X_np.shape[1]
        b = [0]*branchNodeNum
        c = [0]*leafNodeNum
        a = np.zeros((branchNodeNum, p), dtype="float32")
        return a, np.asarray(b,dtype="float32"), np.asarray(c, dtype="float32")

    else:
        model = model.fit(X_np, Y_np)
        p = X.shape[1]
        a, b, c, ab0indList = regTreeWarmStart(model,treeDepth)
        a_all = np.eye(p, dtype="float32")[a]                   
        a_all[ab0indList] = 0

        return a_all, np.asarray(b,dtype="float32"), np.asarray(c, dtype="float32")



