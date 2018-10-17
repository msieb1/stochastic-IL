import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict, defaultdict
from os.path import join
import os
import pickle
from ipdb import set_trace
from networks import VAE

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "1,2"

SAVE_PATH = '../data'

TRAIN_FILES = ['01051_a.pkl', '01051_b.pkl']
VAL_FILES = ['01051_c.pkl']

USE_CUDA = True

def to_var(x, use_cuda=True, volatile=False):
    x = torch.Tensor(x)
    if use_cuda:
        x = x.cuda()
    return x


def main(args):

    ts = time.time()

    datasets = OrderedDict()

    train_set_x = []
    train_set_y = []
    val_set_x = []
    val_set_y = []
    for file in TRAIN_FILES:
        with open(join(SAVE_PATH, file), 'rb') as f:
            loaded_trajectory = pickle.load(f) 
            for state_tuple in loaded_trajectory:       
                train_set_x.extend(state_tuple['action'])
                train_set_y.extend(state_tuple['state_aug'])
    for file in VAL_FILES:
        with open(join(SAVE_PATH, file), 'rb') as f:
            loaded_trajectory = pickle.load(f) 
            for state_tuple in loaded_trajectory:       
                val_set_x.extend(state_tuple['action'])
                val_set_y.extend(state_tuple['state_aug'])

    datasets['train'] = TensorDataset(torch.Tensor(train_set_x), torch.Tensor(train_set_y))
    datasets['val'] = TensorDataset(torch.Tensor(val_set_x), torch.Tensor(val_set_y))

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

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)


    tracker_global = defaultdict(torch.FloatTensor)
    tracker_global['loss'] = 0
    tracker_global['it'] = 0

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

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

                # for i, yi in enumerate(y.data):
                #     # set_trace()
                #     id = len(tracker_epoch)
                #     tracker_epoch[id]['x'] = z[i, 0].data[0]
                #     tracker_epoch[id]['y'] = z[i, 1].data[0]
                #     tracker_epoch[id]['label'] = yi[0]
                #     pass

                loss = loss_fn(recon_x, x, mean, log_var)

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # tracker_global['loss'] = torch.cat((tracker_global['loss'], loss.data/x.size(0)))
                # tracker_global['it'] = torch.cat((tracker_global['it'], torch.Tensor([epoch*len(data_loader)+iteration])))

                if iteration % args.print_every == 100 or iteration == len(data_loader)-1:
                    print("Batch %04d/%i, Loss %9.4f"%(iteration, len(data_loader)-1, loss.data[0]))


                    # if args.conditional:
                    #     c=to_var(torch.arange(0,10).long().view(-1,1))
                    #     x = vae.inference(n=c.size(0), c=c)
                    # else:
                    #     x = vae.inference(n=10)

                    # plt.figure()
                    # plt.figure(figsize=(5,10))
                    # for p in range(10):
                    #     plt.subplot(5,2,p+1)
                    #     if args.conditional:
                    #         plt.text(0,0,"c=%i"%c.data[p][0], color='black', backgroundcolor='white', fontsize=8)
                    #     plt.imshow(x[p].view(28,28).data.numpy())
                    #     plt.axis('off')


                    # if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                    #     if not(os.path.exists(os.path.join(args.fig_root))):
                    #         os.mkdir(os.path.join(args.fig_root))
                    #     os.mkdir(os.path.join(args.fig_root, str(ts)))

                    # plt.savefig(os.path.join(args.fig_root, str(ts), "E%iI%i.png"%(epoch, iteration)), dpi=300)
                    # plt.clf()
                    # plt.close()
        if (epoch + 1) % 4 == 0:
            torch.save(vae.state_dict(), '/home/msieb/projects/CVAE/trained_weights_2/epoch_{}.pk'.format(epoch))
            # df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
            # g = sns.lmplot(x='x', y='y', hue='label', data=df.groupby('label').head(100), fit_reg=False, legend=True)
            # g.savefig(os.path.join(args.fig_root, str(ts), "E%i-Dist.png"%epoch), dpi=300)
            print("saving weights...")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[3, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 3])
    parser.add_argument("--latent_size", type=int, default=10)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional",type=bool, default=True)

    args = parser.parse_args()

main(args)