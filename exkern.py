import numpy as np
import torch
import torch.nn.functional as F
from gpytorch.module import Module

__all__ = ['ElementwiseExKern', 'ExReLU', 'ExErf']


class ElementwiseExKern(Module): #Parameterized
    def K(self, cov, var1, var2=None):
        raise NotImplementedError

    def Kdiag(self, var):
        raise NotImplementedError

    def nlin(self, x):
        """
        The nonlinearity that this is computing the expected inner product of.
        Used for testing.
        """
        raise NotImplementedError


class ExReLU(ElementwiseExKern):
    # TODO : remove name, not used
    def __init__(self, exponent=1, multiply_by_sqrt2=False, name=None):
        super(ExReLU, self).__init__() #name=name
        self.multiply_by_sqrt2 = multiply_by_sqrt2
        if exponent in {0, 1}:
            self.exponent = exponent
        else:
            raise NotImplementedError

    def K(self, cov, var1, var2=None):
        if var2 is None:
            sqrt1 = sqrt2 = torch.sqrt(var1)
        else:
            sqrt1, sqrt2 = torch.sqrt(var1), torch.sqrt(var2)

        norms_prod = sqrt1[:, None, ...] * sqrt2
        norms_prod = torch.reshape(norms_prod, cov.size())#tf.reshape(norms_prod, tf.shape(cov)) #.size()

        cos_theta = torch.clamp(cov / norms_prod, -.9999, .9999) #tf.clip_by_value
        theta = torch.acos(cos_theta)  # angle wrt the previous RKHS #tf.acos

        if self.exponent == 0:
            return .5 - theta/(2*np.pi)

        sin_theta = torch.sqrt(1. - cos_theta**2)
        J = sin_theta + (np.pi - theta) * cos_theta
        if self.multiply_by_sqrt2:
            div = np.pi
        else:
            div = 2*np.pi
        return norms_prod / div * J

    def Kdiag(self, var):
        if self.multiply_by_sqrt2:
            if self.exponent == 0:
                return torch.ones_like(var)
            else:
                return var
        else:
            if self.exponent == 0:
                return torch.ones_like(var)/2
            else:
                return var/2

    def nlin(self, x):
        if self.multiply_by_sqrt2:
            if self.exponent == 0:
                return ((1 + torch.sign(x))/np.sqrt(2))
            elif self.exponent == 1:
                return torch.nn.functional.relu(x) * np.sqrt(2) #tf.nn.relu
        else:
            if self.exponent == 0:
                return ((1 + torch.sign(x))/2)
            elif self.exponent == 1:
                return torch.nn.functional.relu(x)


class ExErf(ElementwiseExKern):
    """The Gaussian error function as a nonlinearity. It's very similar to the
    tanh. Williams 1997"""
    def K(self, cov, var1, var2=None):
        if var2 is None:
            t1 = t2 = 1+2*var1
        else:
            t1, t2 = 1+2*var1, 1+2*var2
        vs = torch.reshape(t1[:, None, ...] * t2, cov.size())
        sin_theta = 2*cov / torch.sqrt(vs)
        return (2/np.pi) * torch.asin(sin_theta)

    def Kdiag(self, var):
        v2 = 2*var
        return (2/np.pi) * torch.asin(v2 / (1 + v2))

    def nlin(self, x):
        return torch.erf(x)
