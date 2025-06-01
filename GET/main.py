import torch
from .treeFunc import readTreePath
from .warmStart import CARTRegWarmStart
from .GET_IterAlp import  multiStartTreeOptbyGRAD_withC
from .subtreePolish import RT


class BaseGETRegressor:
    def __init__(self, treeDepth=4, epochNum=3000, startNum=10, device='cpu'):
        self.treeDepth = treeDepth
        self.epochNum = epochNum
        self.device = torch.device(device)
        self.indices_flags_dict = readTreePath(treeDepth, device)
        self.startNum = startNum
        self.tree = None

    def _prepare_input(self, X):
        if isinstance(X, torch.Tensor):
            return X.clone().detach().to(dtype=torch.float32, device=self.device)
        return torch.tensor(X, dtype=torch.float32, device=self.device)

    def predict(self, X):
        X = self._prepare_input(X)
        n, _ = X.shape
        Tb = 2 ** self.treeDepth - 1
        t = torch.ones(n, device=self.device, dtype=torch.long)

        for _ in range(self.treeDepth):
            decisions = (self.tree['a'][t - 1] * X).sum(dim=1) > self.tree['b'][t - 1]
            t = torch.where(decisions, 2 * t + 1, 2 * t).long()

        Yhat = self.tree['c'][(t - (Tb + 1)).long()]
        return Yhat.detach()


class GETRegressor(BaseGETRegressor):
    def fit(self, X, y):
        X = self._prepare_input(X)
        y = self._prepare_input(y)

        a_init, b_init, c_init = CARTRegWarmStart(X, y, self.treeDepth, self.device)
        cart_warmStart_dict = {"a": a_init, "b": b_init, "c": c_init}
        warmStart = [cart_warmStart_dict]

        _, self.tree = multiStartTreeOptbyGRAD_withC(
            X, y, self.treeDepth, self.indices_flags_dict, self.epochNum, self.device, warmStart, self.startNum
        )
    

class GETSubPolRegressor(BaseGETRegressor):
    def fit(self, X, y):
        X = self._prepare_input(X)
        y = self._prepare_input(y)
        
        _, self.tree, _, _, _ = RT(X, y, self.treeDepth, self.indices_flags_dict, self.epochNum, self.device, self.startNum)