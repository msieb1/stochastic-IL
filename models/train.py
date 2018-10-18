import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
# import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict, defaultdict
from os.path import join
import os
import pickle
import random
from ipdb import set_trace
from networks import VAE
import argparse
from os.path import join
import matplotlib.pyplot as plt
from copy import deepcopy as copy
from logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--encoder_layer_sizes", type=list, default=[3, 128,256])
parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 128, 3])
parser.add_argument("--latent_size", type=int, default=30)
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--fig_root", type=str, default='figs')
parser.add_argument("--conditional",type=bool, default=True)
parser.add_argument("-cu", "--cuda_visible_devices",type=str, default="1,2")
parser.add_argument('-e', '--expname', type=str, required=True)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_visible_devices

BASE_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-2])
EXP_PATH = join(BASE_DIR, 'experiments/{}'.format(args.expname))
SAVE_PATH = join(EXP_PATH, 'data')
MODEL_PATH = join(EXP_PATH, 'trained_weights')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
LOG_PATH = join('tb', args.expname)
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

PKL_FILES = [i for i in os.listdir(SAVE_PATH) if i.startswith('newest')]

USE_CUDA = True

def to_var(x, use_cuda=True, volatile=False):
    x = torch.Tensor(x)
    if use_cuda:
        x = x.cuda()
    return x

def main():

    ts = time.time()

    datasets = OrderedDict()

    train_set_x = []
    train_trajectories  = []
    train_set_y = []
    val_set_x = []
    val_set_y = []
    for file in PKL_FILES:
        with open(join(SAVE_PATH, file), 'rb') as f:
            loaded_trajectory = pickle.load(f) 
            for state_tuple in loaded_trajectory:       
                train_set_x.extend(state_tuple['action'])
                train_set_y.extend(state_tuple['state_aug'])
    print len(train_set_x)
    
    num_val = int(len(train_set_x) * 0.2)
    num_train = len(train_set_x) - num_val
    val_set_x = train_set_x[num_train : ]
    val_set_y = train_set_y[num_train : ]
    train_set_x = train_set_x[: num_train]
    train_set_y = train_set_y[: num_train]

    # Visualize trajectories in 2d:
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')    
    ind = np.random.choice(len(train_set_y), min(len(train_set_y), 5000), replace=False)  
    for ii, pt in enumerate(np.array(train_set_y)[ind]):
        plt.scatter(pt[0], pt[1], s=0.2,c='b')
    plt.savefig(os.path.join(EXP_PATH, '{}_traj_plot.pdf'.format(args.expname)))
    # plt.show()
    # set_trace()
    # return

    datasets['train'] = TensorDataset(torch.Tensor(train_set_x), torch.Tensor(train_set_y))
    datasets['val'] = TensorDataset(torch.Tensor(val_set_x), torch.Tensor(val_set_y))
    logger = Logger(LOG_PATH)

    def loss_fn(recon_x, x, mean, log_var):
        #BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, size_average=False)
        BCE = criterion(recon_x, x)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return BCE + KLD
    criterion = torch.nn.MSELoss()
    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels= 6 if args.conditional else 0
        )
    if USE_CUDA:
        vae = vae.cuda()

    optimizer = torch.optim.SGD(vae.parameters(), lr=args.learning_rate, momentum=0.9)
    tot_iteration = 0
    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))
        print('-'*10)
        print("Epoch: {}".format(epoch + 1))
        for split, dataset in datasets.items():
            if split == 'val':
                print('validation:')
            data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=split=='train')
            for iteration, (x, y) in enumerate(data_loader):

                # set_trace()
                x = to_var(x)[:, :3] * 100
                y = to_var(y)

                x = x.view(-1, 3)
                y = y.view(-1, 6)

                if args.conditional:
                    recon_x, mean, log_var, z = vae(x, y)
                else:
                    recon_x, mean, log_var, z = vae(x)

                loss = loss_fn(recon_x, x, mean, log_var)

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if tot_iteration % 100 == 0:
                        logger.scalar_summary('train_loss', loss.data.item(), tot_iteration)
                else:
                    if tot_iteration % 100 == 0:
                        logger.scalar_summary('val_loss', loss.data.item(), tot_iteration)

                if iteration % args.print_every == 100 or iteration == len(data_loader)-1:
                    print("Batch {0:04d}/{1} Loss {2:9.4f}".format(iteration, len(data_loader)-1, loss.data.item()))
                tot_iteration += 1

        if epoch and epoch % 5 == 0:
            torch.save(vae.state_dict(), join(MODEL_PATH, 'epoch_{}.pk'.format(epoch)))
            print("saving weights...")

if __name__ == '__main__':

    main()
