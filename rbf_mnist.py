

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch import nn
import torch
import gpytorch
import math
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Subset

from torch.distributions import Categorical

from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods.likelihood import Likelihood

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

batch_size = 64
n = 1000
n_epochs = 10
lr = 0.1


class SoftmaxLikelihood(Likelihood):
    """
    Implements the Softmax (multiclass) likelihood used for GP classification.
    """

    def __init__(self, num_features, n_classes, mixing_weights_prior=None):
        super(SoftmaxLikelihood, self).__init__()
        self.num_features = num_features
        self.n_classes = n_classes
        self.register_parameter(
            name="mixing_weights",
            parameter=torch.nn.Parameter(torch.ones(n_classes, num_features).fill_(1.0 / num_features)),
        )
        if mixing_weights_prior is not None:
            self.register_prior("mixing_weights_prior", mixing_weights_prior, "mixing_weights")

    def forward(self, latent_func):
        if not isinstance(latent_func, MultivariateNormal):
            raise RuntimeError(
                "SoftmaxLikelihood expects a multi-variate normally distributed latent function to make predictions"
            )

        n_samples = settings.num_likelihood_samples.value()
        samples = latent_func.rsample(sample_shape=torch.Size((n_samples,)))
        if samples.dim() == 2:
            samples = samples.unsqueeze(-1).transpose(-2, -1)
        samples = samples.permute(1, 2, 0).contiguous()  # Now n_featuers, n_data, n_samples
        if samples.ndimension() != 3:
            raise RuntimeError("f should have 3 dimensions: features x data x samples")
        num_features, n_data, _ = samples.size()
        if num_features != self.num_features:
            raise RuntimeError("There should be %d features" % self.num_features)

        mixed_fs = self.mixing_weights.matmul(samples.view(num_features, n_samples * n_data))
        softmax = torch.nn.functional.softmax(mixed_fs.t(), 1).view(n_data, n_samples, self.n_classes)
        return Categorical(probs=softmax.mean(1))

    def variational_log_probability(self, latent_func, target):
        n_samples = settings.num_likelihood_samples.value()
        samples = latent_func.rsample(sample_shape=torch.Size((n_samples,)))
        if samples.dim() == 2:
            samples = samples.unsqueeze(-1).transpose(-2, -1)
        samples = samples.permute(1, 2, 0).contiguous()  # Now n_featuers, n_data, n_samples
        if samples.ndimension() != 3:
            raise RuntimeError("f should have 3 dimensions: features x data x samples")
        num_features, n_data, _ = samples.size()
        if num_features != self.num_features:
            raise RuntimeError("There should be %d features" % self.num_features)

        mixed_fs = self.mixing_weights.matmul(samples.view(num_features, n_samples * n_data))
        log_prob = -torch.nn.functional.cross_entropy(
            mixed_fs.t(), target.unsqueeze(1).repeat(1, n_samples).view(-1), reduction="sum"
        )
        return log_prob.div(n_samples)


# In[3]:


trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])


# # MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=trans,
                               download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=trans)


train_dataset = Subset(train_dataset, range(n))
test_dataset = Subset(test_dataset, range(1000))

print(len(train_dataset))
print(len(test_dataset))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size ,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)



class GaussianProcessLayer(gpytorch.models.AdditiveGridInducingVariationalGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        super(GaussianProcessLayer, self).__init__(grid_size=grid_size, grid_bounds=[grid_bounds],
                                                   num_dim=num_dim, mixing_params=False, sum_output=False)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class DKLModel(gpytorch.Module):
    def __init__(self, num_dim, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

    def forward(self, x):
        features = x
        features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        res = self.gp_layer(features)
        return res

num_classes = 10
model = DKLModel(num_dim=784).cuda()
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, n_classes=num_classes).cuda()



optimizer = SGD([
    {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
    {'params': model.gp_layer.variational_parameters()},
    {'params': likelihood.parameters()},
], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)

def train(epoch):
    model.train()
    likelihood.train()

    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))

    train_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.reshape(len(target),28*28).cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = -mll(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: %d [%03d/%03d], Loss: %.6f' % (epoch, batch_idx + 1, len(train_loader), loss.item()))

def test():
    from torch.optim import SGD, Adam
    from torch.optim.lr_scheduler import MultiStepLR
    import torch.nn.functional as F
    from torch import nn
    import torch
    model.eval()
    likelihood.eval()

    correct = 0
    for data, target in test_loader:
        data, target = data.reshape(len(target),28*28).cuda(), target.cuda()
        with torch.no_grad():
            output = likelihood(model(data))
            pred = output.probs.argmax(1)
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    print('Test set: Accuracy: {}/{} ({}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
    ))



import time
t = time.time()

for epoch in range(1, n_epochs + 1):
    print("epoch",epoch)
    scheduler.step()
    with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_preconditioner_size(0):
        train(epoch)
    print(time.time() - t)
test()
print(time.time() - t)
