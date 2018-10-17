import time
import argparse
import sys
import numpy as np
import os
from PIL import Image
from pdb import set_trace
import torch
import torch.nn as nn


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "1,2"


def to_var(x, device, volatile=False):
    x = torch.Tensor(x)
    x = x.to(device)
    return x

class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, conditional=False, num_labels=0):
        super(VAE, self).__init__()
        if conditional:
            assert num_labels > 0
        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, k=None):
        batch_size = x.size(0)
        means, log_var = self.encoder(x, k)
        std = torch.exp(0.5 * log_var).cuda()
        # eps = to_var(torch.randn([batch_size, self.latent_size]))
        eps = torch.randn([batch_size, self.latent_size]).cuda()
        z = eps * std + means
        recon_x = self.decoder(z, k)
        return recon_x, means, log_var, z

    def inference(self, use_cuda=True, n=1, k=None):
        batch_size = n
        # z = to_var(torch.randn([batch_size, self.latent_size]))
        if use_cuda:
            z = torch.randn([batch_size, self.latent_size]).cuda()
        else:
            z = torch.randn([batch_size, self.latent_size])
        recon_x = self.decoder(z, k)
        return recon_x

class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):
        super(Encoder, self).__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate( zip(layer_sizes[:-1], layer_sizes[1:]) ):
            self.MLP.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A%i"%(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, k=None):
        if self.conditional:
            # c = idx2onehot(c, n=10)
            
            x = torch.cat((x, k), dim=1)

        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):
        super(Decoder, self).__init__()

        self.MLP = nn.Sequential()
        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate( zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A%i"%(i), module=nn.ReLU())
            else:
                # self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
                self.MLP.add_module(name="tanh", module=nn.Tanh())

    def forward(self, z, k):
        if self.conditional:
            # c = idx2onehot(c, n=10)
            z = torch.cat((z, k), dim=1)

        x = self.MLP(z)
        return x

def vae_loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return BCE + KLD

