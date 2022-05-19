import torch

from PoseEstimation import PoseEstimation
from GradLM import GradLM


def check_grad(x, y):
    params = torch.DoubleTensor(6).fill_(0)
    pe = PoseEstimation(x, y, init_params=params)
    solver = GradLM(y=y.flatten(), func=pe)
    f = solver.optimize()
    return f.params.sum()


xy = [0, 0]
xy[0] = torch.rand(2, 3).double()
xy[1] = torch.DoubleTensor(xy[0].shape).fill_(1)
xy[0].requires_grad = True
assert torch.autograd.gradcheck(check_grad, (xy))

xy = [0, 0]
xy[1] = torch.rand(2, 3).double()
xy[0] = torch.DoubleTensor(xy[1].shape).fill_(1)
xy[1].requires_grad = True
assert torch.autograd.gradcheck(check_grad, (xy))

print("Tests passed!")
