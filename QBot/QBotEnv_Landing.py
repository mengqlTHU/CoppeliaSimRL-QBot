import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium import spaces, logger
from scipy.spatial.transform import Rotation as R
import time

import sys
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from QBotSimModel import QBotSimModel
import math
import random

def cmp(a,b):
    if a>b:
        return 1
    elif a==b:
        return 0
    else:
        return -1

class QBotEnv_Landing(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, port=23000):
        super(QBotEnv_Landing, self).__init__()
        self.target_pos = [0.0, 1.0, 0.64]
        self.target_R = R.from_quat([0, 0, 0, 1])
        self.max_torque = 0.05
        self.port = port

        self.pos_error_thresh = 0.03
        self.angle_error_thresh = 0.1
        self.v_thresh = 0.002
        self.w_thresh = 0.02
        self.touch_ground = False
        self.v_last = np.array([0.0, 0.0, 0.0])
        self.error_angle_last = 100

        self.sim_per_step = 10

        self.target_speed_max = 8 * np.pi
        self.cm_w_max = 2*np.pi
        self.cm_v_max = 1
        self.cm_p_max = 2
        self.cm_theta_max = 2*np.pi

        high = [self.target_speed_max]*3 + [self.cm_w_max]*3 + [self.cm_v_max]*3 \
        + [self.cm_p_max]*3 + [self.cm_theta_max]*3

        high = np.array(
            high,
            dtype=np.float32,
        )

        client = RemoteAPIClient(port=port)
        self.sim = client.require('sim')
        self.sim.setStepping(True)

        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()
        self.state = np.zeros((15,))
        self.setSimInit()

        self.qbot_sim_model = QBotSimModel()
        self.qbot_sim_model.initializeSimModel(self.sim)

        self.sim.startSimulation()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def setSimInit(self):
        self.state[0] = random.uniform(-self.target_speed_max, self.target_speed_max)
        self.state[1] = random.uniform(-self.target_speed_max, self.target_speed_max)
        self.state[2] = random.uniform(-self.target_speed_max, self.target_speed_max)
        self.state[3] = random.uniform(-0.001,0.001)
        self.state[4] = 0.006 + random.uniform(-0.001,0.001)
        self.state[5] = random.uniform(-0.001,0.001)
        self.state[6] = random.uniform(-0.1, 0.1)
        self.state[7] = random.uniform(-0.1, 0.1)
        self.state[8] = random.uniform(-0.1, 0.1)
        self.state[9] = -0.12
        self.state[10] = 0.5
        self.state[11] = -0.36
        self.state[12] = random.uniform(-np.pi, np.pi)
        self.state[13] = random.uniform(-np.pi, np.pi)
        self.state[14] = random.uniform(-np.pi, np.pi)

        robotBase = self.sim.getObject('/lower_hemisphere')
        cm_R = R.from_euler('xyz',self.state[12:15])
        cm_quat = cm_R.as_quat()
        self.sim.setObjectPose(robotBase, np.concatenate([np.array(self.target_pos) - self.state[9:12],cm_quat]).tolist())
        self.sim.setObjectFloatParam(robotBase, self.sim.shapefloatparam_init_velocity_x, self.state[3])
        self.sim.setObjectFloatParam(robotBase, self.sim.shapefloatparam_init_velocity_y, self.state[4])
        self.sim.setObjectFloatParam(robotBase, self.sim.shapefloatparam_init_velocity_z, self.state[5])
        self.sim.setObjectFloatParam(robotBase, self.sim.shapefloatparam_init_velocity_a, self.state[6])
        self.sim.setObjectFloatParam(robotBase, self.sim.shapefloatparam_init_velocity_b, self.state[7])
        self.sim.setObjectFloatParam(robotBase, self.sim.shapefloatparam_init_velocity_g, self.state[8])

    def step(self, action):

        sim_time = self.sim.getSimulationTime()

        # set action
        # Rotate to target speed
        if sim_time < 2:
            self.qbot_sim_model.setJointTargetVelocity('joint0', 100, self.state[0])
            self.qbot_sim_model.setJointTargetVelocity('joint1', 100, self.state[1])
            self.qbot_sim_model.setJointTargetVelocity('joint2', 100, self.state[2])
            cm_pose = self.qbot_sim_model.getCMPose()
            cm_R = R.from_quat(cm_pose[3:])
            rel_R = cm_R.inv() * self.target_R
            self.error_angle_last = np.linalg.norm(rel_R.as_rotvec())



        while sim_time < 2:
            self.sim.step()
            sim_time = self.sim.getSimulationTime()

        #Control Simulation
        robotBase = self.sim.getObject('/lower_hemisphere')
        robotCollection = self.sim.createCollection(0)
        self.sim.addItemToCollection(robotCollection, self.sim.handle_tree, robotBase, 0)
        floor = self.sim.getObject('/Floor')
        result, dist, coll = self.sim.checkDistance(robotCollection, floor, 0.005)
        done = False
        if result == 0:
            self.qbot_sim_model.setJointTargetVelocity('joint0', self.max_torque, action[0] * 8)
            self.qbot_sim_model.setJointTargetVelocity('joint1', self.max_torque, action[1] * 8)
            self.qbot_sim_model.setJointTargetVelocity('joint2', self.max_torque, action[2] * 8)
            for i in range(self.sim_per_step):
                self.sim.step()
        else:
            done = True
            self.touch_ground = True


        # Get State
        v = np.array([0.0, 0.0, 0.0])
        v[0] = self.qbot_sim_model.getJointVelocity('joint0')
        v[1] = self.qbot_sim_model.getJointVelocity('joint1')
        v[2] = self.qbot_sim_model.getJointVelocity('joint2')

        cm_pose = self.qbot_sim_model.getCMPose()
        cm_v, cm_w = self.qbot_sim_model.getCMVelocity()
        cm_v = np.array(cm_v)
        cm_w = np.array(cm_w)
        rel_t = np.array(self.target_pos) - np.array(cm_pose[0:3])

        cm_R = R.from_quat(cm_pose[3:])
        rel_R = cm_R.inv() * self.target_R
        rel_eul = rel_R.as_euler('xyz')

        error_angle = np.linalg.norm(rel_R.as_rotvec())

        #Calculate Reward
        reward = 0.0
        if self.touch_ground:
            reward = 10000.0 * math.exp(-20*error_angle)
        else:
            #Attitude Penalty
            reward = 50 * (self.error_angle_last - error_angle)

            self.error_angle_last = error_angle
            # energy penalty
            energy = 0
            for i in range(3):
                if (v[i] * self.v_last[i]) > 0:
                    energy += 0.5 * 6e-4 * abs(v[i] ** 2 - self.v_last[i] ** 2)
                else:
                    energy += 0.5 * 6e-4 * abs(v[i] ** 2 + self.v_last[i] ** 2)
            reward -= energy
            self.v_last = v

        self.state = np.concatenate([v, cm_v, cm_w, rel_t, rel_eul])

        # print(f"{sim_time}:{reward},{action}")
        return self.state.astype(np.float32), reward, done, False, {}
    
    def reset(self, seed=None):
        # print('Reset the environment after {} counts'.format(self.counts))
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(16,))

        self.touch_ground = False
        self.v_last = np.array([0.0, 0.0, 0.0])
        self.error_angle_last = 100

        self.sim.stopSimulation() # stop the simulation
        time.sleep(1) # ensure the coppeliasim is stopped
        self.sim.setStepping(True)

        self.setSimInit()
        self.sim.startSimulation()
        return np.array(self.state, dtype=np.float32), {}
    
    def render(self):
        return None

    def close(self):
        self.sim.stopSimulation() # stop the simulation
        print('Close the environment')
        return None

if __name__ == "__main__":
    env = QBotEnv()
    env.reset()

    for _ in range(500):
        action = env.action_space.sample() # random action
        env.step(action)
        print(env.state)

    env.close()
