from typing import Any, Optional, Tuple
from torch.autograd import Function
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F


class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

if __name__ == '__main__':
    x = torch.tensor([1., 2., 3.], requires_grad=True)
    y = torch.tensor([4., 5., 6.], requires_grad=True)
    grl = GRL()

    z = torch.pow(x, 2) + torch.pow(y, 2)
    f = z + x + y
    s = 6 * f.sum()

    s = grl(s)
    print(s)
    print(x)
    s.backward()
    print(x.grad)
