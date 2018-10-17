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
np.set_printoptions(precision=4)

xlow = 0.4
xhigh = 0.7
ylow = -0.2
yhigh = 0.2
zlow = 0.3
zhigh = 0.6

SAVE_PATH = '/home/msieb/projects/CVAE/data'

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--runname', type=str, required=True)
args = parser.parse_args()

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
	with open(join(savepath, '{0:05d}_{1}.pkl'.format(int(seqname), args.runname)), 'wb') as f:
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
		# motorsIds.append(env._p.addUserDebugParameter("posX",-dv,dv,0))
		# motorsIds.append(env._p.addUserDebugParameter("posY",-dv,dv,0))
		# motorsIds.append(env._p.addUserDebugParameter("posZ",-dv,dv,0))
		# motorsIds.append(env._p.addUserDebugParameter("yaw",-dv,dv,0))
		# motorsIds.append(env._p.addUserDebugParameter("fingerAngle",0,0.3,.3))
		all_trajectories = []
		n = 0
		while True:
			done = False
			# Reset z to 0,2 higher than intended because it adds +0.2 internally (possibly finger?)
			start = np.array([uf(xlow+0.03, xhigh-0.03), uf(ylow+0.03, yhigh-0.03), uf(zlow+0.03,zhigh-0.03)])
			goal = np.array([uf(xlow+0.03, xhigh-0.03), uf(ylow+0.03, yhigh-0.03), uf(zlow+0.03,zhigh-0.03)])
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
				state, reward, done, info = env.step2(action)
				state = np.array(state)
				#obs = env.getExtendedObservation()
				if ii % 1 == 0:
					# print('normed executed action: {}'.format((state[:3] - state_old[:3])/np.linalg.norm(state[:3] - state_old[:3])))
					# print('executed action:{}'.format(state[:3] - state_old[:3]))
					# print("\n")
					# print('current state: {}'.format(state[:3]))
					# print('goal state: {}'.format(goal))
					# print('action: {}'.format(action[:3]))
					pass
				ii += 1

				time.sleep(0.01)
				# set_trace()
				trajectory['action'].append(action)
				trajectory['state_aug'].append(state_old_aug)
				trajectory['next_state_aug'].append(cat([state[:3], goal]))
				if np.linalg.norm(goal - state[:3]) < eps:
					done = True
				if ii >= 1000:
					break
			if done:
				all_trajectories.append(trajectory)

				if n % 10 == 0:
					print("collected {} trajectories".format(n+1))


				if n % 50 == 0:
					print("save trajectories")
					save_trajectory(all_trajectories, SAVE_PATH, n+1)
				n += 1
	except KeyboardInterrupt:
		save_trajectory(all_trajectories, SAVE_PATH)

if __name__=="__main__":
    main()
