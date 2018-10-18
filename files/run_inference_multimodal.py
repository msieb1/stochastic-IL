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
BASE_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.append(join(BASE_DIR, 'models'))
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
parser.add_argument("--encoder_layer_sizes", type=list, default=[3, 128,256])
parser.add_argument("--decoder_layer_sizes", type=list, default=[256,128, 3])
parser.add_argument("--latent_size", type=int, default=30)
parser.add_argument('-c', "--ckpt_epoch_num", type=int, default=31)
parser.add_argument("-cu", "--cuda_visible_devices",type=str, default="1,2")
parser.add_argument('-e', '--expname', type=str, required=True)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_visible_devices

EXP_PATH = join(BASE_DIR, 'experiments/{}'.format(args.expname))
SAVE_PATH = join(EXP_PATH, 'data')
MODEL_PATH = join(EXP_PATH, 'trained_weights')

USE_CUDA = True
EPOCH=args.ckpt_epoch_num
model = VAE(encoder_layer_sizes=args.encoder_layer_sizes, latent_size=args.latent_size, decoder_layer_sizes=args.decoder_layer_sizes, conditional=True, num_labels=6)
model.load_state_dict(torch.load(join(MODEL_PATH, 'epoch_{}.pk'.format(EPOCH)), map_location=lambda storage, loc: storage))

if USE_CUDA:
    model = model.cuda()
def normalize(a):
    return a/np.linalg.norm(a)

def main():
    env = KukaGymEnv(renders=True,isDiscrete=False, maxSteps = 10000000)
    motorsIds=[]

    all_trajectories = []
    n = 0
    while True:
        done = False
        # Reset z to 0,2 higher than intended because it adds +0.2 internally (possibly finger?)
        start = np.array([uf(xlow+0.03, xlow+0.034), uf(ylow+0.1, ylow+0.15), uf(zlow+0.03,zlow+0.035)])
        goal = np.array([start[0]+ 0.15, start[1], start[2]-0.2])
        state_, success = np.array(env._reset_positions(start))     #default [-0.100000,0.000000,0.070000]
        state = state_[:3]

        action = normalize(goal - state)*0.001
        eps = 0.01
        action = action.tolist()
        action += [0,0]
        if not success:
            env._reset()
            continue
        # print('diff goal - start: {}'.format(goal - state))
        # print('start state: {}, goal state: {}'.format(state, goal))
        # print('true state: {}'.format(env._get_link_state()[0]))

        # print('normed action: {}'.format((goal - state)/np.linalg.norm(state- goal)))
        # print('action: {}'.format(action[:3]))
        ii = 0

        trajectory = {'action': [], 'state_aug': [], 'next_state_aug': []}
        while (not done):
            action[:3] = normalize(goal - state)*0.005

            state_old = copy(state)
            state_old_aug = cat([state, goal])

            cur_state = torch.Tensor(state_old_aug).cuda().unsqueeze_(0)
            action = model.inference(n=1, k=cur_state) 
            action = action.view(-1,).detach().cpu().numpy() / 100 # * 100 during training for normalizing
            # reshape, detach data, move to cpu, and convert to numpy 
            action = cat([action, np.zeros(2,)])
            state_, reward, done, info = env.step2(action)
            state = np.array(state)
            state = state_[:3]
            #obs = env.getExtendedObservation()
            if ii % 10 == 0:
                # # print('normed executed action: {}'.format((state - state_old[:3])/np.linalg.norm(state - state_old[:3])))
                # # print('executed action:{}'.format(state - state_old[:3]))
                samples = []
                for s in range(100):
                    action = model.inference(n=1, k=cur_state) 
                    action = action.view(-1,).detach().cpu().numpy() / 100 # * 100 during training for normalizing
                    samples.append(action)
                print("variance in action selection: {}".format(np.std(samples, axis=0)))
                print("\n")
                print('current state: {}'.format(state))
                print('goal state: {}'.format(goal))
                print('action: {}'.format(action[:3]))
                # pass
            ii += 1

            time.sleep(0.01)
            # set_trace()
            trajectory['action'].append(action)
            trajectory['state_aug'].append(state_old_aug)
            trajectory['next_state_aug'].append(cat([state, goal]))
            if np.linalg.norm(goal - state) < eps:
                done = True
            if ii > 200:
                break

if __name__=="__main__":
    main()
