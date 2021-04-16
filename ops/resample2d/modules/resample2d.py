from torch.nn.modules.module import Module

from ..functions.resample2d import Resample2dFunction


class Resample2d(Module):
    def __init__(self, kernel_size=1, bilinear=True):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size
        self.bilinear = bilinear

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        return Resample2dFunction.apply(input1_c, input2, self.kernel_size, self.bilinear)
