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

def cmp(a,b):
    if a>b:
        return 1
    elif a==b:
        return 0
    else:
        return -1

class QBotEnv_T(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, port=23000):
        super(QBotEnv_T, self).__init__()
        self.push_force = 0
        self.target_pos = [0.0, 1.0, 0.64]
        self.target_R = R.from_quat([0, 0, 0, 1])
        self.max_torque = 0.2
        self.port = port

        self.error_last_min = 10000
        self.error_last_max = 0
        self.error_last = 1.149
        self.error_init = 1.149
        self.pos_error_thresh = 0.03
        self.angle_error_thresh = 0.1
        self.v_thresh = 0.002
        self.w_thresh = 0.02
        self.reduce_thresh = 0.001
        self.v_last = [0,0,0]

        self.sim_per_step = 10

        self.time_max = 240
        self.error_max = 10
        self.target_speed_max = 8 * np.pi
        self.cm_w_max = 2*np.pi
        self.cm_v_max = 1
        self.cm_p_max = 2
        self.cm_theta_max = 2*np.pi

        high = [self.target_speed_max]*3 + [self.cm_w_max]*3 + [self.cm_v_max]*3 \
        + [self.cm_p_max]*3 + [self.cm_theta_max]*3 + [self.time_max]

        high = np.array(
            high,
            dtype=np.float32,
        )

        client = RemoteAPIClient(port=port)
        self.sim = client.require('sim')
        self.sim.setStepping(True)

        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(16,))
        self.state = np.zeros((16,))
        self.state[10] = self.target_pos[1]
        self.counts = 0
        self.steps_beyond_done = None

        self.sim.startSimulation()

        self.qbot_sim_model = QBotSimModel()
        self.qbot_sim_model.initializeSimModel(self.sim)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        # print(f"now on port:{self.port}")
        v = np.array([0.0, 0.0, 0.0])
        v[0] = self.qbot_sim_model.getJointVelocity('joint0')
        v[1] = self.qbot_sim_model.getJointVelocity('joint1')
        v[2] = self.qbot_sim_model.getJointVelocity('joint2')

        cm_pose = self.qbot_sim_model.getCMPose()
        cm_v, cm_w = self.qbot_sim_model.getCMVelocity()
        cm_v = np.array(cm_v)
        cm_w = np.array(cm_w)
        sim_time = self.sim.getSimulationTime()

        rel_t = np.array(self.target_pos) - np.array(cm_pose[0:3])

        cm_R = R.from_quat(cm_pose[3:])
        rel_R = cm_R.inv() * self.target_R
        rel_eul = rel_R.as_euler('xyz')

        error_now = np.linalg.norm(rel_t)
        error_horizontal = np.linalg.norm(rel_t[0:2])
        error_angle = np.linalg.norm(rel_eul)

        # set action
        # print(f"action is {action}")
        if sim_time < 4.9:
            action[0] = 0
            action[1] = 0
            action[2] = 0
        self.qbot_sim_model.setJointTargetVelocity('joint0', action[0], cmp(action[0], 0) * self.target_speed_max)
        self.qbot_sim_model.setJointTargetVelocity('joint1', action[1], cmp(action[1], 0) * self.target_speed_max)
        self.qbot_sim_model.setJointTargetVelocity('joint2', action[2], cmp(action[2], 0) * self.target_speed_max)

        done = (sim_time >= self.time_max) or ((error_now <= self.pos_error_thresh) and (np.linalg.norm(cm_v) <= self.v_thresh) \
                                               and (error_angle <= self.angle_error_thresh) and (np.linalg.norm(cm_w) <= self.w_thresh)) \
                or (np.linalg.norm(rel_t) > 3) or ((sim_time >= 60)  and (error_horizontal > 0.95))
        done = bool(done)

        # attitude reward
        reward = 0.0
        if error_now < (self.error_last_min - self.reduce_thresh):
            reward = 1.0
            self.error_last_min = error_now
        elif error_now > (self.error_last_max + self.reduce_thresh):
            reward = -1.0
            self.error_last_max = error_now
        # reward = 100 * (self.error_last - error_now)
        # self.error_last = error_now

        # attitude reward
        # if error_horizontal < 0.5 * self.error_init:
        #     reward -= 0.3 * error_angle

        #final reward
        if ((error_now <= self.pos_error_thresh) and (np.linalg.norm(cm_v) <= self.v_thresh) \
            and (error_angle <= self.angle_error_thresh) and (np.linalg.norm(cm_w) <= self.w_thresh)):
            reward = 10000.0
        elif (sim_time >= 60) and (np.linalg.norm(cm_v) <= self.v_thresh) and (np.linalg.norm(cm_w) <= self.w_thresh):
            reward = 10000.0 * math.exp(-50*error_horizontal)
        # elif sim_time >= self.time_max:
        #     reward = (max(0, 25000*(0.1-error_now)) +
        #               max(0, 1e6*(0.0025-np.linalg.norm(cm_v))) +
        #               max(0, 1e5*(0.025-np.linalg.norm(cm_w))))
        else:
            robotBase = self.sim.getObject('/lower_hemisphere')
            robotCollection = self.sim.createCollection(0)
            self.sim.addItemToCollection(robotCollection, self.sim.handle_tree, robotBase, 0)
            floor = self.sim.getObject('/Floor')
            if sim_time > 60:
                result, dist, coll = self.sim.checkDistance(robotCollection, floor, 0.005)
                if result > 0:
                    done = True
                    reward = 10000.0 * math.exp(-50 * error_horizontal)

        # time penalty
        reward = reward - 0.1

        #energy penalty
        # energy = action[0] ** 2+ action[1] ** 2 + action[2] ** 2
        # reward -= energy * 0.1
        energy = 0
        for i in range(3):
            if (v[i] * self.v_last[i]) > 0:
                energy += 0.5 * 6e-4 * abs(v[i] ** 2 - self.v_last[i] ** 2)
            else:
                energy += 0.5 * 6e-4 * abs(v[i] ** 2 + self.v_last[i] ** 2)
        reward -= energy
        self.v_last = v

        self.state = np.concatenate([v, cm_v, cm_w, rel_t, rel_eul, np.array([sim_time])])
        self.counts += 1

        # if sim_time > 5:
        #     pass
        for i in range(self.sim_per_step):
            self.sim.step()
        # print(f"{sim_time}:{reward},{action}")
        return self.state.astype(np.float32), reward, done, False, {}
    
    def reset(self, seed=None):
        # print('Reset the environment after {} counts'.format(self.counts))
        self.counts = 0
        self.push_force = 0
        self.error_last_min = 10000
        self.error_last_max = 0
        self.error_last = 1.149
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(16,))
        self.state = np.zeros((16,))
        self.state[10] = self.target_pos[1]
        self.v_last=[0, 0, 0]
        self.steps_beyond_done = None

        self.sim.stopSimulation() # stop the simulation
        time.sleep(1) # ensure the coppeliasim is stopped
        self.sim.setStepping(True)

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
