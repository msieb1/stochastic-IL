#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
import sys
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
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)

xlow = 0.4
xhigh = 0.7
ylow = -0.2
yhigh = 0.2
zlow = 0.3
zhigh = 0.6

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--expname', type=str, required=True)
parser.add_argument('-r', '--runname', type=str, required=True)
args = parser.parse_args()
BASE_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-2])
EXP_PATH = join(BASE_DIR, 'experiments/{}'.format(args.expname))
SAVE_PATH = join(EXP_PATH, 'data')
print("Saving to {}".format(SAVE_PATH))
MODEL_PATH = join(EXP_PATH, 'trained_weights')

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

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
            seq_names = [int(i.split('.')[0][:5]) for i in os.listdir(savepath)]
            latest_seq = sorted(map(int, seq_names), reverse=True)[0]
            seqname = str(latest_seq+1)
        print('No seqname specified, using: %s' % seqname)
    with open(join(savepath, '{0}_{1}.pkl'.format(seqname, args.runname)), 'wb') as f:
        pickle.dump(file, f)

def main():
    env = KukaGymEnv(renders=False,isDiscrete=False, maxSteps = 10000000)
        
    try:  
        motorsIds=[]
        #motorsIds.append(env._p.addUserDebugParameter("posX",0.4,0.75,0.537))
        #motorsIds.append(env._p.addUserDebugParameter("posY",-.22,.3,0.0))
        #motorsIds.append(env._p.addUserDebugParameter("posZ",0.1,1,0.2))
        #motorsIds.append(env._p.addUserDebugParameter("yaw",-3.14,3.14,0))
        #motorsIds.append(env._p.addUserDebugParameter("fingerAngle",0,0.3,.3))
        
        dv = 0.00
        all_trajectories = []
        n = 0
        while True:
            done = False
            # Reset z to 0,2 higher than intended because it adds +0.2 internally (possibly finger?)
            # start = np.array([uf(xlow+0.03, xhigh-0.03), uf(ylow+0.03, yhigh-0.03), uf(zlow+0.03,zhigh-0.03)])
            # goal = np.array([uf(xlow+0.03, xhigh-0.03), uf(ylow+0.03, yhigh-0.03), uf(zlow+0.03,zhigh-0.03)])

            # start = np.array([uf(xlow+0.03, xlow+0.034), uf(ylow+0.1, ylow+0.15), uf(zlow+0.03,zlow+0.035)])
            start = np.array([xlow+0.032, uf(ylow+0.1, ylow+0.15), uf(zlow+0.02,zlow+0.04)])
            if start[1] < ylow+0.125:
                y_offset = -uf(0.08,0.12)
            else:
                y_offset = uf(0.08,0.12)

            if start[1] < ylow+0.125:
                branching_scale = -uf(0.004, 0.012)
            else:
                branching_scale = +uf(0.004, 0.012)
               
            # switching_point_fraction = 1/uf(3.5,4.5)
            switching_point_fraction = 0.5
            state_, success = np.array(env._reset_positions(start))     #default [-0.100000,0.000000,0.070000]
            state = state_[:3]
            true_start_state = copy(state)
            # Specify agent goals
            goal = np.array([start[0]+ 0.2, start[1], start[2]])
            # set_trace()
            wp_goal = copy(goal)
            # wp_goal[2] = true_start_state[2]
            branched_goal_wp = true_start_state[1] + y_offset

            action = normalize(goal - state)*0.001
            eps = 0.01
            action = action.tolist()
            action += [0,0]

            np.save(join(SAVE_PATH, "start.npy"), start)
            np.save(join(SAVE_PATH, 'goal.npy'), goal)           

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
                # print('state x: {}'.format(state[0]))
                if state[0] < true_start_state[0] + (goal[0] - true_start_state[0]) * switching_point_fraction:
                    #wp_goal = copy(goal)
                    wp_goal[1] += branching_scale
                    #wp_goal[0] = true_start_state[0] +(true_start_state[0]+goal[0])/4.0
                elif state[0] < true_start_state[0] + (goal[0] - true_start_state[0]) * 1:
                    # print('branched off')
                    #wp_goal = copy(goal)
                    #wp_goal[0] = true_start_state[0] + (goal[0] - true_start_state[0]) * 2.0/3.0
                    #wp_goal[1] = true_start_state[1] + y_offset
                    wp_goal[1] = goal[1]
                    wp_goal[0] += 0.01
                    pass
                else:
                    # print('branch back in')
                    wp_goal = copy(goal)
                
                action[:3] = normalize(wp_goal - state)*0.005

                state_old = copy(state)
                state_old_aug = cat([state_old, goal])
                state_, reward, done, info = env.step2(action)
                state = state_[:3]
                state = np.array(state)
                #obs = env.getExtendedObservation()
                if ii % 10 == 0:
                    # print('normed executed action: {}'.format((state - state_old[:3])/np.linalg.norm(state - state_old[:3])))
                    # print('executed action:{}'.format(state - state_old[:3]))
                    # print("\n")
                    # print('current state: {}'.format(state))
                    # print('goal state: {}'.format(goal))
                    # print('action: {}'.format(action[:3]))
                    pass
                ii += 1

                time.sleep(0.01)
                trajectory['action'].append(copy(action))
                trajectory['state_aug'].append(state_old_aug)
                trajectory['next_state_aug'].append(cat([state, goal]))

                if np.linalg.norm(goal - state) < eps:
                    done = True
                if ii >= 1000:
                    break
            if done:
                all_trajectories.append(trajectory)

                if n % 10 == 0:
                    print("collected {} trajectories".format(n+1))

                if n % 50 == 0:
                    # print("save trajectories")
                    save_trajectory(all_trajectories, SAVE_PATH, 'backup')
                if n % 50 == 0:
                    save_trajectory(all_trajectories, SAVE_PATH, 'newest')
                    print("saved trajectories")

                n += 1
    except KeyboardInterrupt:
        pass
        # save_trajectory(all_trajectories, SAVE_PATH)

if __name__=="__main__":
    main()
