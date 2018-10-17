#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
from kukaGymEnvReach import KukaGymEnvReach as KukaGymEnv
import time

from numpy import array
import numpy as np
from copy import deepcopy as copy
from numpy.random import uniform as uf

from pdb import set_trace
import pickle
from os.path import join
from numpy import concatenate as cat
import argparse
import torch
import sys

sys.path.append('../models')
from networks import VAE

np.set_printoptions(precision=4)

xlow = 0.4
xhigh = 0.7
ylow = -0.2
yhigh = 0.2
zlow = 0.3
zhigh = 0.6



parser = argparse.ArgumentParser()
parser.add_argument('--task', dest='task', type=str, default='reach')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--encoder_layer_sizes", type=list, default=[3, 256])
parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 3])
parser.add_argument("--latent_size", type=int, default=10)
parser.add_argument('-e', '--expname', type=str, required=True)

args = parser.parse_args()

EXP_PATH = '../experiments/{}'.format(args.expname)
SAVE_PATH = join(EXP_PATH, 'data')
MODEL_PATH = join(EXP_PATH, 'trained_weights')

USE_CUDA = True
EPOCH=23
model = VAE(encoder_layer_sizes=args.encoder_layer_sizes, latent_size=args.latent_size, decoder_layer_sizes=args.decoder_layer_sizes, conditional=True, num_labels=6)
model.load_state_dict(torch.load(join(MODEL_PATH, 'epoch_{}.pk'.format(EPOCH)), map_location=lambda storage, loc: storage))
if USE_CUDA:
    model = model.cuda()

def normalize(a):
    return a/np.linalg.norm(a)

def save_trajectory(file, savepath, seqname=None):
    if seqname is not None:
        seqname = seqname
    else:
      # If there's no video directory, this is the first sequence.
      if not os.listdir(savepath):
        seqname = '0'
      else:
        # Otherwise, get the latest sequence name and increment it.
        seq_names = [int(i.split('.')[0]) for i in os.listdir(savepath)]
        latest_seq = sorted(map(int, seq_names), reverse=True)[0]
        seqname = str(latest_seq+1)
      print('No seqname specified, using: %s' % seqname)
    with open(join(savepath, '{0:05d}_{1}}.pkl'.format(int(seqname), args.runname)), 'wb') as f:
        pickle.dump(file, f)


def main():
    env = KukaGymEnv(renders=True,isDiscrete=False, maxSteps = 10000000)
    motorsIds=[]

    all_trajectories = []
    n = 0
    while True:
        done = False
        # Reset z to 0,2 higher than intended because it adds +0.2 internally (possibly finger?)
        start = np.array([uf(xlow+0.05, xhigh-0.05), uf(ylow+0.05, yhigh-0.07), uf(zlow+0.05,zhigh-0.05)])
        goal = np.array([uf(xlow+0.05, xhigh-0.05), uf(ylow+0.05, yhigh-0.07), uf(zlow+0.05,zhigh-0.05)])
        state, success = np.array(env._reset_positions(start))     #default [-0.100000,0.000000,0.070000]
        action = normalize(goal - state[:3])*0.001
        eps = 0.01
        action = action.tolist()
        action += [0,0]
        if not success:
            env._reset()
            continue
        # print('diff goal - start: {}'.format(goal - state[:3]))
        # print('start state: {}, goal state: {}'.format(state[:3], goal))
        # print('true state: {}'.format(env._get_link_state()[0]))

        # print('normed action: {}'.format((goal - state[:3])/np.linalg.norm(state[:3]- goal)))
        # print('action: {}'.format(action[:3]))
        ii = 0

        trajectory = {'action': [], 'state_aug': [], 'next_state_aug': []}
        while (not done):
            action[:3] = normalize(goal - state[:3])*0.005

            state_old = copy(state)
            state_old_aug = cat([state[:3], goal])

            cur_state = torch.Tensor(state_old_aug).cuda()
            action = model.inference(n=1, k=cur_state.unsqueeze_(0)) 

            # reshape, detach data, move to cpu, and convert to numpy 
            action = action.view(-1,).detach().cpu().numpy() / 100 # * 100 during training for normalizing
            action = cat([action, np.zeros(2,)])
            state, reward, done, info = env.step2(action)
            state = np.array(state)
            obs = env.getExtendedObservation()
            if ii % 1 == 0:
                # # print('normed executed action: {}'.format((state[:3] - state_old[:3])/np.linalg.norm(state[:3] - state_old[:3])))
                # # print('executed action:{}'.format(state[:3] - state_old[:3]))
                print("\n")
                print('current state: {}'.format(state[:3]))
                print('goal state: {}'.format(goal))
                print('action: {}'.format(action[:3]))
                # pass
            ii += 1

            time.sleep(0.01)
            # set_trace()
            trajectory['action'].append(action)
            trajectory['state_aug'].append(state_old_aug)
            trajectory['next_state_aug'].append(cat([state[:3], goal]))
            if np.linalg.norm(goal - state[:3]) < eps:
                done = True
            if ii > 200:
                break


if __name__=="__main__":
    main()
