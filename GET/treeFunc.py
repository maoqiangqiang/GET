import torch
import math
import h5py
import numpy as np
from typing import Dict, Tuple
import os



# jit version of update_c 
#  tree: Dict[str, torch.Tensor] does not allow any int type value (only tensor tyoe value). 
@torch.jit.script
def update_c(X: torch.Tensor, y: torch.Tensor, treeDepth: int, tree: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    n, p = X.shape
    Tb = 2 ** treeDepth - 1  # branch node size
    Tleaf = 2 ** treeDepth  # leaf node size
    miny = torch.min(y)
    maxy = torch.max(y)

    # Initialize c and z
    c = torch.full([int(Tleaf)], (miny + maxy) / 2, device=X.device)
    z = torch.ones(n, device=X.device, dtype=torch.long)

    # Calculate the path for each data point in a vectorized manner
    for _ in range(treeDepth):
        # decisions = (tree['a'][z - 1] * X).sum(dim=1) >= tree['b'][z - 1]
        decisions = (tree['a'][z - 1] * X).sum(dim=1) > tree['b'][z - 1]                         # since  ax<=b -> left branch in CART.
        z = torch.where(decisions, 2 * z + 1, 2 * z)
    

    # Adjust indices to match the original function
    z = z - (Tb + 1)
    # Ensure z is of dtype int64
    z = z.to(torch.int64)
    # Calculate the mean of y for each unique element in z
    unique_z, counts = torch.unique(z, return_counts=True)
    # print("unique_z: {}".format(unique_z))
    # print("counts: {}".format(counts))
    sums = torch.zeros_like(c).scatter_add_(0, z, y)
    c[unique_z] = sums[unique_z] / counts.float()
    tree['c'] = c
    return tree




@torch.jit.script
def objv_cost(X: torch.Tensor, y: torch.Tensor, treeDepth: int, tree: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    n, p = X.shape
    Tb = 2 ** treeDepth - 1
    t = torch.ones(n, device=X.device, dtype=torch.long)

    for iiii in range(treeDepth):
        # decisions = (tree['a'][t - 1] * X).sum(dim=1) >= tree['b'][t - 1]
        decisions = (tree['a'][t - 1] * X).sum(dim=1) > tree['b'][t - 1]
        t = torch.where(decisions, 2 * t + 1, 2 * t).long()

    Yhat = tree['c'][(t - (Tb + 1)).long()]
    # Manual R2 Score computation
    total_variance = torch.sum((y - torch.mean(y)) ** 2)
    residual_variance = torch.sum((Yhat - y) ** 2)
    r2_score = 1 - (residual_variance / total_variance)

    # return torch.sum((Yhat - y) ** 2)/n, r2_score
    return residual_variance/n, r2_score




## accelerate the getBranchNodes function
def getBranchNodes(ind, prelayer):
    ind -= 1
    branchNodes = [ind]
    currentNodes = [ind]
    for _ in range(prelayer-1):
        nextNodes = [2*node + j for node in currentNodes for j in [1, 2]]
        branchNodes.extend(nextNodes)
        currentNodes = nextNodes
    return branchNodes



## Tree Path Calculation and Saving 
def treePathCalculation(treeDepth, Data_device):
    branchNodeNum = 2 ** (treeDepth) - 1
    leafNodeNum = 2 ** treeDepth  

    ancestorTF_pairs = []              

    for Idx in range(leafNodeNum):
        leafIdx = Idx + branchNodeNum + 1
        log_val = math.floor(math.log2(leafIdx))
        ancestors = [leafIdx >> j for j in range(log_val, -1, -1)]
        ancestors = torch.as_tensor(ancestors, device= Data_device)
        ancestors_shifted = ancestors[1:] + 1
        oddEven = ((-1) ** ancestors_shifted + 1) / 2     # 0 -> I1, 1 -> (1-I1)

        # print("leafIdx: ", leafIdx)
        # print("ancestors: ", ancestors)
        # print("oddEven: ", oddEven)

        ancestorIdxTemp = [(ancestors[ancestorIdx]-1).cpu().numpy().item() for ancestorIdx in range(len(ancestors)-1)]
        oddEvenTemp = [bool(element) for element in ((1- oddEven))]                     # 0/False -> (1-I1), 1/True -> I1

        ancestorTF_zip = list(zip(ancestorIdxTemp, oddEvenTemp))
        ancestorTF_pairs.append(ancestorTF_zip)
    ancestorTF_pairs_np = np.array(ancestorTF_pairs)
    print("ancestorTF_pairs_np: ", ancestorTF_pairs_np.shape)
    print("ancestorTF_pairs_np: ", ancestorTF_pairs_np)

    ## save the ancestorTF_pairs into a HDF5 file
    package_dir = os.path.dirname(__file__)
    ancestor_dir = os.path.join(package_dir, "ancestorTF_File")
    file_path = os.path.join(ancestor_dir, f"ancestorTF_pairs_D{treeDepth}.hdf5")

    with h5py.File(file_path, 'w') as ancestorTF_File:
        # Save the numpy array as a dataset in the HDF5 file
        ancestorTF_File.create_dataset("indicator_pairs", data=ancestorTF_pairs)




## Read Tree Path from HDF5 file
def readTreePath(treeDepth, device):
    ## read the treePath from the HDF5 file
    indices_flags_dict = {}

    package_dir = os.path.dirname(__file__)
    ancestor_dir = os.path.join(package_dir, "ancestorTF_File")

    for treeDepthEach in range(treeDepth):
        treeDepthEach += 1
        file_path = os.path.join(ancestor_dir, f"ancestorTF_pairs_D{treeDepthEach}.hdf5")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing ancestorTF file: {file_path}. Run treePathCalculation() to generate it.")
        
        with h5py.File(file_path, 'r') as ancestorTF_File:
            ancestorTF_pairs = ancestorTF_File['indicator_pairs'][:]
        
        ancestorTF_pairs_tensor = torch.tensor(ancestorTF_pairs, dtype=torch.long, device=device)

        # Create tensors for indices and flags
        indices_tensor_long = ancestorTF_pairs_tensor[..., 0]
        flags_tensor_long = ancestorTF_pairs_tensor[..., 1]

        key = "D"+str(treeDepthEach)
        indices_flags_dict[key] = {
            'indices_tensor': indices_tensor_long,
            'flags_tensor': flags_tensor_long
        }

    # print(indices_flags_dict.keys())
    return indices_flags_dict






if __name__ == "__main__":

    for treeDepth in [1,2,3,4,5,6,7,8,9,10,11,12]:
        Data_device = torch.device("cpu")
        treePathCalculation(treeDepth, Data_device)