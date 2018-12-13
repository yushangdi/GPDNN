import torch
import torch.nn.functional as F
import gpytorch
from gpytorch.kernels import Kernel
from typing import List
import numpy as np
# import abc
import exkern

from exkern import ElementwiseExKern


class DeepKernelBase(Kernel):
    "General kernel for deep networks"
    def __init__(self,
                 input_shape: List[int],
                 block_sizes: List[int],
                 block_strides: List[int],
                 kernel_size: int,
                 recurse_kern: ElementwiseExKern,
                 conv_stride: int = 1,
                 active_dims: slice = None,
                 input_type = None,
                 name: str = None):
        input_dim = np.prod(input_shape)
        super(DeepKernelBase, self).__init__(input_dim, active_dims)

        self.input_shape = list(np.copy(input_shape))
        self.block_sizes = np.copy(block_sizes).astype(np.int32)
        self.block_strides = np.copy(block_strides).astype(np.int32)
        self.kernel_size = kernel_size
        self.recurse_kern = recurse_kern
        self.conv_stride = conv_stride
        if input_type is None:
            input_type = torch.float32
        self.input_type = input_type
        self.register_parameter(name = "var_weight",
        parameter=torch.nn.Parameter(torch.zeros(1)), prior = None)
        self.register_parameter(name = "var_bias",
        parameter=torch.nn.Parameter(torch.zeros(1)), prior = None)

    def forward(self, x1, x2 = None, **params):
        if x2 is None:
            return self.K(x1)
        else:
            if len(x1.size()) == 5:
                x1 = x1[0]
            if len(x2.size()) == 5:
                x2 = x2[0]
            return self.K(x1,x2)

    def K(self, X, X2=None):
        # Concatenate the covariance between X and X2 and their respective
        # variances. Only 1 variance is needed if X2 is None.
        if X.dtype != self.input_type or (
                X2 is not None and X2.dtype != self.input_type):
            raise TypeError("Input dtypes are wrong: {} or {} are not {}"
                            .format(X.dtype, X2.dtype, self.input_type))
        if X2 is None:
            N = N2 = X.size()[0]
            var_z_list = [
                torch.reshape(torch.pow(X,2), [N] + self.input_shape),
                torch.reshape(X[:, None, :] * X, [N*N] + self.input_shape)]

            def apply_recurse_kern(var_a_all, concat_outputs=True):
                var_a_1 = var_a_all[:N]
                var_a_cross = var_a_all[N:]
                vz = [self.recurse_kern.Kdiag(var_a_1),
                      self.recurse_kern.K(var_a_cross, var_a_1, None)]
                if concat_outputs:
                    return torch.cat(vz, 0)
                return vz

        else:
            N, N2 = X.size()[0], X2.size()[0]
            var_z_list = [
                torch.reshape(torch.pow(X,2), [N] + self.input_shape),
                torch.reshape(torch.pow(X2,2), [N2] + self.input_shape),
                torch.reshape(X[:, None, :] * X2, [N*N2] + self.input_shape)]
            cross_start = N + N2

            def apply_recurse_kern(var_a_all, concat_outputs=True):
                var_a_1 = var_a_all[:N]
                var_a_2 = var_a_all[N:cross_start]
                var_a_cross = var_a_all[cross_start:]
                vz = [self.recurse_kern.Kdiag(var_a_1),
                      self.recurse_kern.Kdiag(var_a_2),
                      self.recurse_kern.K(var_a_cross, var_a_1, var_a_2)]
                if concat_outputs:
                    return torch.cat(vz, 0)
                return vz
        inputs = torch.cat(var_z_list, 0)

        if len(self.block_sizes) > 0:
            # Define almost all the network
            inputs = self.headless_network(inputs, apply_recurse_kern)
            # Last nonlinearity before final dense layer
            var_z_list = apply_recurse_kern(inputs, concat_outputs=False)
        # averaging for the final dense layer
        var_z_cross = torch.reshape(var_z_list[-1], [N, N2, -1])
        var_z_cross_last = torch.mean(var_z_cross,2)
        result = F.softplus(self.var_bias) + F.softplus(self.var_weight) * var_z_cross_last
        # .type(torch.FloatTensor)
        # if self.input_type != torch.float64:
        #     print("Casting kernel from {} to {}"
        #           .format(self.input_type, torch.float64))
        #     return result.float()
        return result

    def Kdiag(self, X):
        if X.dtype != self.input_type:
            raise TypeError("Input dtype is wrong: {} is not {}"
                            .format(X.dtype, self.input_type))
        inputs = torch.reshape(torch.pow(X,2), [-1] + self.input_shape)
        if len(self.block_sizes) > 0:
            inputs = self.headless_network(inputs, self.recurse_kern.Kdiag)
            # Last dense layer
            inputs = self.recurse_kern.Kdiag(inputs)
        var_z_last = inputs.copy()
        for i in range(len(inputs.shape)):
            var_z_last = torch.mean(var_z_last,1)
        result = F.softplus(self.var_bias) + F.softplus(self.var_weight) * var_z_last
        # if self.input_type != torch.float64:
        #     print("Casting kernel from {} to {}"
        #           .format(self.input_type, torch.float64))
        #     return result.float()
        return result

    def headless_network(self, inputs, apply_recurse_kern):
        """
        Apply the network that this kernel defines, except the last dense layer.
        The last dense layer is different for K and Kdiag.
        """
        raise NotImplementedError


class DeepKernelTesting(DeepKernelBase):
    """
    Reimplement original DeepKernel to test ResNet
    """
    def headless_network(self, inputs, apply_recurse_kern):
        in_chans = inputs.size()[1]

        W_init = (torch.ones([1, in_chans, self.kernel_size, self.kernel_size])*
                           F.softplus(self.var_weight) / in_chans) #.type(torch.FloatTensor)
        inputs = F.conv2d(inputs, W_init, bias=None, stride=[1,1], padding=0
                          ) + F.softplus(self.var_bias) #.type(torch.FloatTensor)
        W = (torch.ones([1,  1, self.kernel_size, self.kernel_size])*
                    F.softplus(self.var_weight))#.type(torch.FloatTensor)  # No dividing by fan_in
        for _ in range(1, len(self.block_sizes)):
            inputs = apply_recurse_kern(inputs)
            inputs = F.conv2d(inputs, W,
            bias=None, stride=1, padding=0)
            inputs = inputs + F.softplus(self.var_bias)#.type(torch.FloatTensor)
        return inputs
