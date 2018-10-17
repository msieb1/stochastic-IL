import os, inspect

import pybullet as p
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import kuka
import random
import pybullet_data
from pkg_resources import parse_version
from pdb import set_trace
from kukaGymEnv import KukaGymEnv

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class KukaGymEnvReach(KukaGymEnv):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second' : 50
  }

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps = 1000):

    #print("KukaGymEnv __init__")
    self._isDiscrete = isDiscrete
    self._timeStep = 1./240.
    self._timePerAction = 30
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180
    self._cam_pitch = -40
    self.goal_state = None

    self._p = p
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid<0):
         cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3,180,-41,[0.52,-0.2,-0.33])
    else:
      p.connect(p.DIRECT)
    #timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
    self._seed()
    self._reset()
    observationDim = len(self.getExtendedObservation())
    #print("observationDim")
    #print(observationDim)

    observation_high = np.array([largeValObservation] * observationDim)
    if (self._isDiscrete):
      self.action_space = spaces.Discrete(7)
    else:
       action_dim = 3
       self._action_bound = 1
       action_high = np.array([self._action_bound] * action_dim)
       self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)
    self.viewer = None
   

  def getExtendedObservation(self):
     self._observation = self._kuka.getObservation()
     gripperState  = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaGripperIndex)
     gripperPos = gripperState[0]
     gripperOrn = gripperState[1]
     blockPos,blockOrn = p.getBasePositionAndOrientation(self.blockUid)

     invGripperPos,invGripperOrn = p.invertTransform(gripperPos,gripperOrn)
     gripperMat = p.getMatrixFromQuaternion(gripperOrn)
     dir0 = [gripperMat[0],gripperMat[3],gripperMat[6]]
     dir1 = [gripperMat[1],gripperMat[4],gripperMat[7]]
     dir2 = [gripperMat[2],gripperMat[5],gripperMat[8]]

     gripperEul =  p.getEulerFromQuaternion(gripperOrn)
     #print("gripperEul")
     #print(gripperEul)
     blockPosInGripper,blockOrnInGripper = p.multiplyTransforms(invGripperPos,invGripperOrn,blockPos,blockOrn)
     projectedBlockPos2D =[blockPosInGripper[0],blockPosInGripper[1]]
     blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
     #print("projectedBlockPos2D")
     #print(projectedBlockPos2D)
     #print("blockEulerInGripper")
     #print(blockEulerInGripper)

     #we return the relative x,y position and euler angle of block in gripper space
     blockInGripperPosXYEulZ =[blockPosInGripper[0],blockPosInGripper[1],blockEulerInGripper[2]]

     #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir0[0],gripperPos[1]+dir0[1],gripperPos[2]+dir0[2]],[1,0,0],lifeTime=1)
     #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir1[0],gripperPos[1]+dir1[1],gripperPos[2]+dir1[2]],[0,1,0],lifeTime=1)
     #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir2[0],gripperPos[1]+dir2[1],gripperPos[2]+dir2[2]],[0,0,1],lifeTime=1)
     return self._observation

  def _reset(self, reset_pos=None):
      #print("KukaGymEnv _reset")
    self.terminated = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])

    p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)

    xpos = 0.55 +0.12*random.random()
    ypos = 0 +0.2*random.random()
    ang = 3.14*0.5+3.1415925438*random.random()
    orn = p.getQuaternionFromEuler([0,0,ang])
    self.blockUid =p.loadURDF(os.path.join(self._urdfRoot,"block.urdf"), xpos,ypos,-0.15,orn[0],orn[1],orn[2],orn[3])
    p.setGravity(0,0,-10)
    self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)

    self._envStepCounter = 0
    p.stepSimulation()
    if reset_pos is not None:
      # self._kuka.moveKukaEndtoPos(reset_pos)
      self._kuka.endEffectorPos = reset_pos
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def _get_link_state(self):
    true_state = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaEndEffectorIndex)
    return true_state

  def _reset_positions(self, reset_pos):
    xpos = 0.55 +0.12*random.random()
    ypos = 0 +0.2*random.random()
    ang = 3.14*0.5+3.1415925438*random.random()
    orn = p.getQuaternionFromEuler([0,0,ang])
    #self.blockUid =p.loadURDF(os.path.join(self._urdfRoot,"block.urdf"), xpos,ypos,-0.15,orn[0],orn[1],orn[2],orn[3])
    p.setGravity(0,0,-10)

    self._envStepCounter = 0

    # self._kuka.moveKukaEndtoPos(reset_pos)
    self._kuka.endEffectorPos = reset_pos
    self._kuka.endEffectorPos[2] -= 0.2 #finger tip adjustment in z
    success = self._kuka.moveKukaEndtoPos(reset_pos)
    self.endEffectorPos = self._get_link_state()[0]


    self._observation = self.getExtendedObservation()
    return self._observation, success
