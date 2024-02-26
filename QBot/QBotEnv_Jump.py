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

class QBotEnv_Jump(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, port=23000):
        super(QBotEnv_Jump, self).__init__()
        self.push_force = 0
        self.target_pos = [0.0, 1.0, 0.64]
        self.target_R = R.from_quat([0, 0, 0, 1])
        self.max_torque = 0.05
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
        self.v_last = np.array([0.0,0.0,0.0])
        self.cm_v_last = np.array([0.0, 0.0, 0.0])
        self.leave_ground = False

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

        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32)
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

        sim_time = self.sim.getSimulationTime()

        # set action
        # print(f"action is {action}")
        if sim_time < 4.9:
            self.qbot_sim_model.setJointTargetVelocity('joint0', 0, 0)
            self.qbot_sim_model.setJointTargetVelocity('joint1', 0, 0)
            self.qbot_sim_model.setJointTargetVelocity('joint2', 0, 0)

        while sim_time < 4.9:
            self.sim.step()
            sim_time = self.sim.getSimulationTime()

        #Control Simulation
        robotBase = self.sim.getObject('/lower_hemisphere')
        robotCollection = self.sim.createCollection(0)
        self.sim.addItemToCollection(robotCollection, self.sim.handle_tree, robotBase, 0)
        floor = self.sim.getObject('/Floor')
        result, dist, coll = self.sim.checkDistance(robotCollection, floor, 0.05)
        done = False
        if not self.leave_ground and result==1:
            self.qbot_sim_model.setJointTargetVelocity('joint0', self.max_torque, action[0] * 8)
            self.qbot_sim_model.setJointTargetVelocity('joint1', self.max_torque, action[1] * 8)
            self.qbot_sim_model.setJointTargetVelocity('joint2', self.max_torque, action[2] * 8)
            self.sim.step()
        else:
            self.leave_ground = True


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

        error_now = np.linalg.norm(rel_t)
        error_horizontal = np.linalg.norm(rel_t[0:2])
        error_angle = np.linalg.norm(rel_eul)

        #Calculate Reward
        reward = 0.0
        if self.leave_ground:
            result, dist, coll = self.sim.checkDistance(robotCollection, floor, 0.005)
            while result==0:
                for i in range(self.sim_per_step):
                    self.sim.step()
                sim_time = self.sim.getSimulationTime()

                cm_pose = self.qbot_sim_model.getCMPose()
                rel_t = np.array(self.target_pos) - np.array(cm_pose[0:3])
                error_horizontal = np.linalg.norm(rel_t[0:2])

                if (sim_time >= self.time_max) or (np.linalg.norm(rel_t) > 3) or ((sim_time >= 60)  and (error_horizontal > 0.95)):
                    done = True
                    break
                result, dist, coll = self.sim.checkDistance(robotCollection, floor, 0.005)
            if done:
                reward = -100.0
            else:
                done = True
                reward = 10000.0 * math.exp(-5 * error_horizontal)
        else:
            #Accelaration Penalty
            acc = cm_v - self.cm_v_last
            acc_h = acc[0:2]
            acc_hnorm = acc_h / np.linalg.norm(acc_h)
            rel_t_hnorm = rel_t[0:2] / np.linalg.norm(rel_t[0:2])

            reward = 10000 * np.dot(acc_h, rel_t_hnorm) - abs(np.arccos(np.dot(acc_hnorm, rel_t_hnorm)))
            self.cm_v_last = cm_v

            # energy penalty
            energy = 0
            for i in range(3):
                if (v[i] * self.v_last[i]) > 0:
                    energy += 0.5 * 6e-4 * abs(v[i] ** 2 - self.v_last[i] ** 2)
                else:
                    energy += 0.5 * 6e-4 * abs(v[i] ** 2 + self.v_last[i] ** 2)
            reward -= energy

            self.v_last = v


        self.state = np.concatenate([v, cm_v, cm_w, rel_t, rel_eul, np.array([sim_time])])

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
        self.v_last = np.array([0.0, 0.0, 0.0])
        self.cm_v_last = np.array([0.0, 0.0, 0.0])
        self.steps_beyond_done = None
        self.leave_ground = False

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
