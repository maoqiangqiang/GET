import torch 
import copy 
import math

from .treeFunc import objv_cost, update_c

from .modifiedScheduler import ChainedScheduler


import os 
#  enforce deterministic behavior in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"


class scaledSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(input)
        ctx.scale = scale
        sigmoid = torch.sigmoid(scale * input)
        ctx.sigmoid = sigmoid
        return sigmoid

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        scale = ctx.scale
        sigmoid = ctx.sigmoid
        # sigmoid = torch.sigmoid(scale * input)
        gradient = grad_output * (scale * sigmoid * (1 - sigmoid))
        return gradient, None


class branchNodeNet(torch.nn.Module):
    def __init__(self, treeDepth: int, p: int, scale: float) -> None:
        super().__init__()
        self.depth = treeDepth
        self.featNum = p
        # self.treesize = 2 ** (self.depth + 1) - 1
        self.branchNodeNum = 2 ** (self.depth) - 1
        self.scale = scale

        self.linear1 = torch.nn.Linear(self.featNum, self.branchNodeNum)
        self.scaledSigmoid = scaledSigmoid.apply 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.scaledSigmoid(-x, self.scale)
        return x



@torch.jit.script
def objectiveFuncwithC(branchOutput: torch.Tensor, Y_train: torch.Tensor, c_leafLable: torch.Tensor,  treeDepth: int, indices_tensor: torch.Tensor, flags_tensor: torch.Tensor) -> torch.Tensor: 

    indicators_complement = 1 - branchOutput
    Y_train_diff = (Y_train.view(-1, 1) - c_leafLable.view(1, -1)).pow_(2)
    indicators_stack = torch.stack((indicators_complement, branchOutput))
    selected_indicators = indicators_stack[flags_tensor, :, indices_tensor]
    indicator_pairs = selected_indicators.prod(dim=1).transpose(0, 1)
    objv = (indicator_pairs * Y_train_diff).sum()
    N_num = Y_train.shape[0]
    objv = objv / N_num 
    return objv




def initialize_parameters(choice, net, warmStarts, Y_train, device):

    a, b, c = None, None, None
    if choice < len(warmStarts) and warmStarts[choice] is not None:
        # print("warm start")
        # print("choice is {}".format(choice))
        ws = warmStarts[choice]
        a, b, c = ws["a"], ws["b"], ws["c"]

    else:
        # print("random init")
        a = torch.rand(net.linear1.weight.shape,dtype=torch.float32, device=device) * (2.0)+(-1.0)
        b = torch.rand(net.linear1.bias.shape, dtype=torch.float32, device=device) * (2.0)+(-1.0)
        c = torch.rand([2 ** net.depth], dtype=torch.float32, device=device) * (torch.max(Y_train) - torch.min(Y_train)) + torch.min(Y_train)

    return a, b, c


class callbackFuncs:
    def CalMSEonEpochEnd(self, X_train, Y_train, treeDepth, net, c_leafLable):
        with torch.no_grad():    
            # Clone the weights and biases to avoid modifying the original tensors
            a_grad = net.linear1.weight.clone().detach()  # Ensuring no gradients
            b_grad = net.linear1.bias.clone().detach() * (-1.0)
            c_grad = c_leafLable.clone().detach()  # Ensuring no gradients
            treeEpoch = {"a": a_grad, "b": b_grad, "c": c_grad}
        treeEpoch = update_c(X_train, Y_train, treeDepth, treeEpoch)
        objvMSE_Epoch, r2_Epoch = objv_cost(X_train, Y_train, treeDepth, treeEpoch)
        return objvMSE_Epoch, r2_Epoch, treeEpoch
        



def treeOptbyGRADwithC(treeDepth, indices_flags_dict, epochNum, X_train, Y_train, device, warmStarts, scaleFactor, lrscheList,  idxStart, callback):

    ## hyperparameters
    learningRate, T0, warmupsteps, gamma = lrscheList[0], lrscheList[1], lrscheList[2], lrscheList[3]

    ##  net
    p = X_train.shape[1]
    scale = torch.tensor([scaleFactor], device=device)
    net = branchNodeNet(treeDepth, p, scale).to(device, non_blocking=True)

    ### initialize weight and bias of net
    a, b, c = initialize_parameters(idxStart, net, warmStarts, Y_train, device)
    customWeigt = torch.as_tensor(a, dtype=torch.float32, device=device)
    customBias = torch.as_tensor(b, dtype=torch.float32, device=device)
    net.linear1.weight = torch.nn.Parameter(customWeigt)
    net.linear1.bias = torch.nn.Parameter(customBias)
    ## optimized parameters
    c_leafLable = torch.as_tensor(c, dtype=torch.float32, device=device).requires_grad_()

    ## Optimizer 
    # optimizer = torch.optim.Adam(list(net.parameters())+[c_leafLable], lr=learningRate)
    optimizer = torch.optim.AdamW(list(net.parameters())+[c_leafLable], lr=learningRate)

    # ############################################
    # ########### if to disable the scheduler, must comment the following code; otherwise, the lr will be affected ############ 
    # restartNum = 1
    # # Sn = 2**restartNum-1
    # Sn = restartNum
    # warmupsteps = 10
    # T0= math.ceil((epochNum - warmupsteps)/Sn)
    # # print("T0 is {} and warmupsteps is {}".format(T0, warmupsteps))
    # epochNumNew = T0*Sn+warmupsteps
    # # print("epochNumNew is {}".format(epochNumNew))
    # gamma = (1/restartNum)**(1/(restartNum-1))   if restartNum != 1 else 1
    # # print("gamma is {}".format(gamma))
    scheduler = ChainedScheduler(optimizer, T_0=T0, T_mul=1, eta_min=1e-6, max_lr=learningRate, warmup_steps=warmupsteps, gamma=gamma)

    # load the indices_tensor and flags_tensor from the indices_flags_dict
    indices_tensor = indices_flags_dict["D"+str(treeDepth)]["indices_tensor"]
    flags_tensor = indices_flags_dict["D"+str(treeDepth)]["flags_tensor"]


    objvMSE_EpochBest = float('inf')
    r2_EpochBest = float('-inf')
    tree_EpochBest = None

    for epoch in range(epochNum):

        optimizer.zero_grad(set_to_none=True)   

        # Forward pass with a batch
        branchOutput = net(X_train)
        objv = objectiveFuncwithC(branchOutput, Y_train, c_leafLable, treeDepth, indices_tensor, flags_tensor)

        ## check the real mse loss of the current tree
        objvMSE_Epoch, r2_Epoch, treeEpoch = callback.CalMSEonEpochEnd(X_train, Y_train, treeDepth, net, c_leafLable)
        if objvMSE_Epoch < objvMSE_EpochBest:
            objvMSE_EpochBest = objvMSE_Epoch
            r2_EpochBest = r2_Epoch
            tree_EpochBest = copy.deepcopy(treeEpoch)

        # Backward pass and optimize
        objv.backward()

        optimizer.step()
        scheduler.step()

    return objvMSE_EpochBest, r2_EpochBest, tree_EpochBest






def multiStartTreeOptbyGRAD_withC(X_train, Y_train, treeDepth, indices_flags_dict, epochNum, device, warmStarts, startNum):

    objvmin = 1e10
    treeOpt = None


    ############################################
    ########### if to disable the scheduler, must comment the following code; otherwise, the lr will be affected ############ 
    restartNum = 1
    # Sn = 2**restartNum-1
    Sn = restartNum
    warmupsteps = 10
    T0= math.ceil((epochNum - warmupsteps)/Sn)
    # print("T0 is {} and warmupsteps is {}".format(T0, warmupsteps))
    epochNumNew = T0*Sn+warmupsteps
    # print("epochNumNew is {}".format(epochNumNew))
    gamma = (1/restartNum)**(1/(restartNum-1))   if restartNum != 1 else 1
    # print("gamma is {}".format(gamma))
    lrscheList = [0.01, T0, warmupsteps, gamma]
    ############################################
    
    cartWarmStart = warmStarts[0]
    treeCART = {"a": torch.tensor(cartWarmStart["a"], device=device), "b": torch.tensor(-cartWarmStart["b"], device=device), "c": torch.tensor(cartWarmStart["c"], device=device)}
    objvMSECART, r2CART = objv_cost(X_train, Y_train, treeDepth, treeCART)


    # callback function
    callback = callbackFuncs()


    if len(warmStarts) > startNum:
        startNum = len(warmStarts)
    # print("startNum is {}".format(startNum))
    for idxStart in range(startNum):
        scaleStart = torch.rand(1, device=device) * 20 + 5
        scaleEnd = torch.rand(1, device=device) * 40 + 80
        scaleList = [scaleStart.item(), scaleEnd.item()]


        ObjvAlp = 1e10
        bestTreeAlp = None
        warmStarts_cur = [warmStarts[idxStart]] if idxStart < len(warmStarts) else  [None]
        idxCurr = 0
        

        for scaleFactor in scaleList:

            objvCurr, r2Curr, treeCurrent = treeOptbyGRADwithC(treeDepth, indices_flags_dict, epochNum, X_train, Y_train, device, warmStarts_cur, scaleFactor, lrscheList, idxCurr, callback)
            if objvCurr < ObjvAlp:
                ObjvAlp = objvCurr
                bestTreeAlp = copy.deepcopy(treeCurrent)
            warmStarts_cur.append( {"a": treeCurrent["a"], "b": treeCurrent["b"]* (-1.0), "c": treeCurrent["c"]})
            idxCurr += 1

        if ObjvAlp < objvMSECART:
            # TreeAfterGrad = copy.deepcopy(bestTreeAlp)
            TreeAfterGrad = bestTreeAlp
        else:
            # TreeAfterGrad = copy.deepcopy(treeCART)
            TreeAfterGrad = treeCART
            ObjvAlp = objvMSECART
        
        if ObjvAlp < objvmin:
            objvmin = ObjvAlp
            treeOpt = copy.deepcopy(TreeAfterGrad)

    return objvmin, treeOpt
























