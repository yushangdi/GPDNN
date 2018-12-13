from typing import List

from exkern import ElementwiseExKern, ExReLU, ExErf

import gpytorch
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import SequentialSampler, BatchSampler


class DeepKernel(gpytorch.kernels.Kernel):
    """
    General deep kernel for n-dimensional convolutional networks
    Should be superseded by dkern.DeepKernelTesting, but that doesn't
    necessarily work yet.
    """
    def __init__(self,
                 input_shape: List[int],
                 filter_sizes: List[List[int]],
                 recurse_kern: ElementwiseExKern,
                 pooling_layers: List[int],
                 pooling_sizes: List[List[int]],
                 var_weight: float = 1.0,
                 var_bias: float = 1.0,
                 padding: List[int] = 0,
                 strides: List[List[int]] = None,
                 data_format: str = "NCHW",
                 active_dims: slice = None,
                 skip_freq: int = -1,
                 name: str = None):
        input_dim = np.prod(input_shape)

        super(DeepKernel, self).__init__(input_dim, active_dims)

        self.filter_sizes = np.copy(filter_sizes).astype(np.int32)
        self.n_layers = len(filter_sizes)
        self.input_shape = list(np.copy(input_shape))
        self.recurse_kern = recurse_kern
        self.skip_freq = skip_freq

        inferred_data_format = "NC" + "DHW"[4-len(input_shape):]
        if inferred_data_format != data_format:
            raise ValueError(("Inferred and supplied data formats "
                              "inconsistent: {} vs {}")
                             .format(data_format, inferred_data_format))
        self.data_format = data_format

        if not isinstance(padding, list):
            self.padding = [padding] * len(self.filter_sizes)
        else:
            self.padding = padding
        if len(self.padding) != len(self.filter_sizes):
            raise ValueError(("Mismatching number of layers in `padding` vs "
                              "`filter_sizes`: {} vs {}").format(
                                  len(self.padding), len(self.filter_sizes)))

        if strides is None:
            self.strides = np.ones([self.n_layers, len(input_shape)-1],
                                   dtype=np.int32)
        else:
            self.strides = np.copy(strides).astype(np.int32)
        if len(self.strides) != self.n_layers:
            raise ValueError(("Mismatching number of layers in `strides`: "
                              "{} vs {}").format(
                                  len(self.strides), self.n_layers))

        ############################## finished
        self.register_parameter(name = "var_weight",
        parameter=torch.nn.Parameter(torch.ones(1).type(torch.FloatTensor)), prior = None)
        self.register_parameter(name = "var_bias",
        parameter=torch.nn.Parameter(torch.ones(1).type(torch.FloatTensor)), prior = None)

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
        #print("Shape of You by Ed Sheeran", X.shape)
        if X2 is None:
            N = N2 = X.size()[0]
            var_z_list = [
                # tf.reshape(X**2, [N] + self.input_shape),
                ####### BUG ######### ELEMENTWISE
                (X**2).view([N] + self.input_shape),
                # tf.reshape(X[:, None, :] * X, [N*N] + self.input_shape)
                (X[:,None,:] * X).view([N*N] + self.input_shape)
                ]
            cross_start = N
        else:
            N, N2 = X.size()[0], X2.size()[0]
            var_z_list = [
                (X**2).view([N] + self.input_shape),
                (X2**2).view([N2] + self.input_shape),
                (X[:, None, :] * X2).view([N*N2] + self.input_shape)]
            cross_start = N + N2
        # mean over C if NCHW
        var_z_list = [torch.mean(z, dim=1, keepdim=True)
                      for z in var_z_list]
        #print("shape of varzlist, not you", var_z_list[0].shape)
        var_z_previous = None

        for i in range(self.n_layers):
            # Do the convolution for all the co/variances at once
            var_z = torch.cat(var_z_list, dim=0) # (2,1,4,4),(2*2,1,4,4) => (6,1,4,4)
            if (i > 0 and ((isinstance(self.skip_freq, list) and i in self.skip_freq) or
                           (self.skip_freq > 0 and i % self.skip_freq == 0))):
                var_z = var_z + var_z_previous
                var_z_previous = var_z
            elif i == 0:
                # initialize var_z_previous
                var_z_previous = var_z
            var_a_all = self.lin_step(i, var_z)
            # if i == 0:
            #print("shape of var a all lalalaa", var_a_all.shape)

            # Disentangle the output of the convolution and compute the next
            # layer's co/variances
            var_a_cross = var_a_all[cross_start:]
            if X2 is None:
                var_a_1 = var_a_all[:N]
                var_z_list = [self.recurse_kern.Kdiag(var_a_1),
                              self.recurse_kern.K(var_a_cross, var_a_1, None)]
            else:
                var_a_1 = var_a_all[:N]
                var_a_2 = var_a_all[N:cross_start]
                var_z_list = [self.recurse_kern.Kdiag(var_a_1),
                              self.recurse_kern.Kdiag(var_a_2),
                              self.recurse_kern.K(var_a_cross, var_a_1, var_a_2)]
        # The final layer
        var_z_cross = (var_z_list[-1]).view([N,N2,-1])
        var_z_cross_last = torch.mean(var_z_cross, dim=2)

        return self.var_bias + self.var_weight * var_z_cross_last


    def Kdiag(self, X):
        X_sq = (X**2).view([-1] + self.input_shape)
        var_z = torch.mean(X_sq, dim=1, keepdim=True)
        for i in range(self.n_layers):
            var_a = self.lin_step(i, var_z)
            var_z = self.recurse_kern.Kdiag(var_a)

        all_except_first = np.arange(1, len(var_z.shape))
        var_z_last = torch.mean(var_z, dim=all_except_first)
        return self.var_bias + self.var_weight * var_z_last


    def lin_step(self, i, x):
        if len(x.shape) == 2:
            a = self.var_weight * x
        else:
            # https://www.tensorflow.org/api_docs/python/tf/nn/convolution
            # https://pytorch.org/docs/stable/nn.html#torch-nn-functional
            f = torch.ones(tuple([1, 1] + list(self.filter_sizes[i]))).type(torch.FloatTensor)*self.var_weight
            #f = torch.ones(tuple(list(self.filter_sizes[i]) + [1, 1])).type(torch.FloatTensor)*self.var_weight
            a = F.conv2d(
                x, f, stride=tuple(self.strides[i]), padding=self.padding[i])
        return a + self.var_bias


    def get_Wb(self, i, X_shape=None, n_samples=None, n_filters=None):
        "Unlike the kernel, this operates in NHWC"
        try:
            if self._W[i] is not None and self._b[i] is not None:
                return self._W[i], self._b[i]
        except AttributeError:
            self._W, self._b = ([None]*(self.n_layers + 1) for _ in 'Wb')
        try:
            std_b = self._std_b
        except AttributeError:
            std_b = self._std_b = self.var_bias**0.5

        if i == self.n_layers:  # Final weights and biases
            final_dim = np.prod(list(map(int, X_shape[1:])))
            shape_W = [n_samples, final_dim, n_filters]
            shape_b = [n_samples, n_filters]
            std_W = (self.var_weight / final_dim) ** 0.5
        else:
            if i == 0:
                fan_in = int(X_shape[-1])
            else:
                fan_in = n_filters
            fs = list(self.filter_sizes[i])
            shape_W = [n_samples] + fs + [fan_in, n_filters]
            shape_b = [n_samples] + [1]*len(fs) + [n_filters]
            std_W = (self.var_weight / fan_in) ** 0.5

        # torch transformation without name
        self._W[i] = torch.randn(shape_W)*std_W
        # torch transformation without name
        self._b[i] = torch.randn(shape_b)*std_b
        return self._W[i], self._b[i]

    def get_average_pooling_filter(self, i, X_shape=None, n_samples=None, n_filters=None):

        if i == self.n_layers:  # Final weights and biases
            final_dim = np.prod(list(map(int, X_shape[1:])))
            shape_W = [n_samples, final_dim, n_filters]
            shape_b = [n_samples, n_filters]
            std_W = (self.var_weight / final_dim) ** 0.5
        else:
            if i == 0:
                fan_in = int(X_shape[-1])
            else:
                fan_in = n_filters
            ps = list(self.pooling_sizes[i])
            shape_W = [n_samples] + ps + [fan_in, n_filters]

        W_pooling = torch.ones(shape_W)
        return W_pooling


    def fast_1sample_equivalent_BNN(self, X, Ws=None, bs=None):
        if Ws is None or bs is None:
            Ws, bs = (list(t[0] for t in t_list) for t_list in [self._W, self._b])
        batch = X.size()[0]
        for W, b, st, pd in zip(Ws[:-1], bs[:-1], self.strides, self.padding):
            b_reshaped = b.view([1, -1, 1, 1])
            strides = [1, 1] + list(st)
            X = F.conv2d(X, W, stride=strides, padding=pd) + b_reshaped
            X = self.recurse_kern.nlin(X)
        return X.view([batch, -1]) @ Ws[-1] + bs[-1]

    def extract_image_patches(imgs,ksizes,strides,rates,paddings,name=None):
        size = imgs.Size()
        imgs = imgs.view((size[0],size[1],-1),-1)
        imgs = imgs.unfold(2,ksizes[1]*ksizes[2],ksizes[1]*strides[1])
        slice = list(filter(lambda x: x != 'inf', list(map(lambda iv: iv[1] if (iv[0] % strides[2] > ksizes[2]) else 'inf', enumerate(b)))))
        imgs = imgs[:,slice,:strides[1]//ksizes[1],:].contiguous().view(-1,ksizes[1]*ksizes[2]*sizes[3])
        imgs = torch.cat(tuple(imgs),dim=1).unfold(1,ksizes[1]*ksizes[2]*sizes[3],ksizes[1]*ksizes[2]*sizes[3]).unsqueeze(0)


    def equivalent_BNN(self, X, n_samples, n_filters=128):
        if list(map(int, X.shape)) != [1] + self.input_shape:
            raise NotImplementedError("Can only deal with 1 input image")

        # Unlike the kernel, this function operates in NHWC. This is because of
        # the `extract_image_patches` function
        tp_order = np.concatenate([[0], np.arange(2, len(X.shape)), [1]])
        X = torch.transpose(torch.transpose(X, 1, 2),2,3)  # NCHW -> NHWC

        # The name of the first dimension of the einsum. In the first linear
        # transform, it should be "a", to broadcast the "n" dimension of
        # samples of parameters along it. In all other iterations it should be
        # "n".
        first = 'a'
        batch_dim = 1

        for i in range(self.n_layers):


            W, b = self.get_Wb(i, X.size(), n_samples, n_filters)
            equation = "{first:}{dims:}i,nij->n{dims:}j".format(
                first=first, dims="dhw"[4-len(self.input_shape):])
            if i in pooling_layers:
                if len(self.pooling_sizes[i]) == 0:
                    Xp = X
                elif len(self.pooling_sizes[i]) == 2:
                    h, w = self.pooling_sizes[i]
                    Xp = extract_image_patches(
                        X, [1, h, w, 1], [1, h, w, 1], [1, 1, 1, 1],0)
                else:
                    raise NotImplementedError("convolutions other than 2d")
                W_pooling = self.get_average_pooling_filter(i, X.size(), n_samples, n_filters)
                W_pooling_flat_in = W_pooling.view([n_samples, -1, W_pooling.size()[-1]])
                X = torch.einsum(equation, [Xp, W_pooling_flat_in])

            if len(self.filter_sizes[i]) == 0:
                Xp = X
            elif len(self.filter_sizes[i]) == 2:
                h, w = self.filter_sizes[i]
                sh, sw = self.strides[i]
                Xp = extract_image_patches(
                    X, [1, h, w, 1], [1, sh, sw, 1], [1, 1, 1, 1],
                    self.padding[i])
            else:
                raise NotImplementedError("convolutions other than 2d")
            # We're explicitly doing the convolution by extracting patches and
            # a multiplication, so this flatten is needed.
            W_flat_in = W.view([n_samples, -1, W.size()[-1]])
            X = self.recurse_kern.nlin(torch.einsum(equation, [Xp, W_flat_in]) + b)
            first = 'n'  # Now we have `n_samples` in the batch dimension
            batch_dim = n_samples

        W, b = self.get_Wb(self.n_layers, X.shape, n_samples, 1)
        X_flat = X.view([batch_dim, -1])
        Wx = torch.einsum("{first:}i,nij->nj".format(first=first), [X_flat, W])
        return Wx + b



class ZeroMeanGauss(gpytorch.priors.torch_priors.NormalPrior):
    def __init__(self, var):
        gpytorch.priors.prior.Prior.__init__(self)
        self.loc = 0.0
        self.scale = var

    def logp(self, x):
        c = np.log(2*np.pi) + np.log(self.scale)
        return -.5 * (c*torch.Size(x)
                      + torch.sum((x**2)/self.scale))


class ConvNet(gpytorch.models.GP):
    "L2-regularised ConvNet as a Model"
    def __init__(self, X, Y, kern, minibatch_size=None, n_filters=256, name: str = None):
        super(ConvNet, self).__init__(name=name)
        if not hasattr(kern, 'W_'):
            # Create W_ and b_ as attributes in kernel
            X_zeros = np.zeros([1] + kern.input_shape)
            _ = kern.equivalent_BNN(
                X=torch.zeros([1] + kern.input_shape),
                n_samples=1,
                n_filters=n_filters)
        self._kern = kern

        # Make MiniBatches if necessary
        if minibatch_size is None:
            self.train_inputs = X
            self.train_targets = Y
            self.scale_factor = 1.
        else:
            self.train_inputs = torch.Tensor(list(BatchSampler(SequentialSampler(X), batch_size=minibatch_siz, drop_last=False)))
            self.train_targets = torch.Tensor(list(BatchSampler(SequentialSampler(Y), batch_size=minibatch_siz, drop_last=False)))
            self.scale_factor = X.shape[0] / minibatch_size
        self.n_labels = int(np.max(Y)+1)

        # Create GPFlow parameters with the relevant size of the network
        Ws, bs = [], []
        for i, (W, b) in enumerate(zip(kern._W, kern._b)):
            if i == kern.n_layers:
                W_shape = [int(W.shape[1]), self.n_labels]
                b_shape = [self.n_labels]
            else:
                W_shape = list(map(int, W.shape[1:]))
                b_shape = [n_filters]
            W_var = kern.var_weight.read_value()/W_shape[-2]
            b_var = kern.var_bias.read_value()
            W_init = np.sqrt(W_var) * np.random.randn(*W_shape)
            b_init = np.sqrt(b_var) * np.random.randn(*b_shape)
            Ws.append(W_init) #, prior=ZeroMeanGauss(W_var)))
            bs.append(b_init) #, prior=ZeroMeanGauss(b_var)))
        # self.Ws = gpflow.params.ParamList(Ws)
        # self.bs = gpflow.params.ParamList(bs)
        self.register_parameter(name='Ws',parameter=torch.nn.Parameter(Ws),prior=None)
        self.register_parameter(name='bs',parameter=torch.nn.Parameter(bs),prior=None)

    def _build_objective(self, likelihood_tensor, prior_tensor):
        return self.scale_factor * likelihood_tensor - prior_tensor  # likelihood_tensor is already a loss

    def _build_likelihood(self):
        # Get around fast_1sample_equivalent_BNN not getting tensors from param
        Ws_tensors = list(self.Ws[i] for i in range(len(self.Ws)))
        bs_tensors = list(self.bs[i] for i in range(len(self.bs)))
        logits = self._kern.fast_1sample_equivalent_BNN(
            self.X.view([-1] + self._kern.input_shape),
            Ws=Ws_tensors, bs=bs_tensors)
        # return tf.losses.sparse_softmax_cross_entropy(labels=self.Y, logits=logits)
        return F.nll_loss(F.softmax(logits), self.Y)

    def predict_y(self, Xnew):
        return self._build_predict_y(Xnew), torch.Tensor([0])[0]

    def _build_predict_y(self, Xnew):
        Ws_tensors = list(self.Ws[i] for i in range(len(self.Ws)))
        bs_tensors = list(self.bs[i] for i in range(len(self.bs)))
        logits = self._kern.fast_1sample_equivalent_BNN(
            Xnew.view([-1] + self._kern.input_shape),
            Ws=Ws_tensors, bs=bs_tensors)
        return F.softmax(logits)


if __name__ == '__main__':
    import deep_ckern as dk, numpy as np
    k = dk.DeepKernel([1, 16, 16], [[3, 3]]*5, dk.ExReLU())
    X = np.random.randn(3, 16**2)
    k.compute_K(X, X)
