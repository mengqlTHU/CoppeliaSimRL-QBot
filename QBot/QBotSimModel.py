import sys
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# CartPole simulation model for VREP
class QBotSimModel():
    def __init__(self, name='CartPole'):
        """
        :param: name: string
            name of objective
        """
        super(self.__class__, self).__init__()
        self.name = name
        self.client_ID = None
        self.sim = None

        self.joint0_handle = None
        self.joint1_handle = None
        self.joint2_handle = None
        self.cm_handle = None

    def initializeSimModel(self, sim):

        self.sim = sim

        self.joint0_handle = self.sim.getObjectHandle('Revolute_joint0')
        print('get object joint0 ok.')

        self.joint1_handle = self.sim.getObjectHandle('Revolute_joint1')
        print('get object joint1 ok.')

        self.joint2_handle = self.sim.getObjectHandle('Revolute_joint2')
        print('get object joint2 ok.')

        self.cm_handle = self.sim.getObjectHandle('cm')

        # Get the joint position
        # q = vrep_sim.simxGetJointPosition(self.client_ID, self.prismatic_joint_handle, vrep_sim.simx_opmode_streaming)
        # q = vrep_sim.simxGetJointPosition(self.client_ID, self.revolute_joint_handle, vrep_sim.simx_opmode_streaming)

    
    def getJointPosition(self, joint_name):
        """
        :param: joint_name: string
        """
        q = 0
        if joint_name == 'joint0':
            q = self.sim.getJointPosition(self.joint0_handle)
        elif joint_name == 'joint1':
            q = self.sim.getJointPosition(self.joint1_handle)
        elif joint_name == 'joint2':
            q = self.sim.getJointPosition(self.joint2_handle)
        else:
            print('Error: joint name: \' ' + joint_name + '\' can not be recognized.')

        return q

    def getJointVelocity(self, joint_name):
        """
        :param: joint_name: string
        """
        v = 0
        if joint_name == 'joint0':
            v = self.sim.getJointVelocity(self.joint0_handle)
        elif joint_name == 'joint1':
            v = self.sim.getJointVelocity(self.joint1_handle)
        elif joint_name == 'joint2':
            v = self.sim.getJointVelocity(self.joint2_handle)
        else:
            print('Error: joint name: \' ' + joint_name + '\' can not be recognized.')

        return v

    def setJointPosition(self, joint_name, pos):
        """
        :param: joint_name: string
        """
        if joint_name == 'joint0':
            self.sim.setJointPosition(self.joint0_handle, pos)
        elif joint_name == 'joint1':
            self.sim.setJointPosition(self.joint1_handle, pos)
        elif joint_name == 'joint2':
            self.sim.setJointPosition(self.joint2_handle, pos)
        else:
            print('Error: joint name: \' ' + joint_name + '\' can not be recognized.')

        return 0

    def setJointTargetVelocity(self, joint_name, torque, target_vel):

        if joint_name == 'joint0':
            self.sim.setJointTargetVelocity(self.joint0_handle, target_vel)
            self.sim.setJointMaxForce(self.joint0_handle, abs(torque))
        elif joint_name == 'joint1':
            self.sim.setJointTargetVelocity(self.joint1_handle, target_vel)
            self.sim.setJointMaxForce(self.joint1_handle, abs(torque))
        elif joint_name == 'joint2':
            self.sim.setJointTargetVelocity(self.joint2_handle, target_vel)
            self.sim.setJointMaxForce(self.joint2_handle, abs(torque))
        else:
            print('Error: joint name: \' ' + joint_name + '\' can not be recognized.')

    def getCMPose(self):

        return self.sim.getObjectPose(self.cm_handle)

    def getCMVelocity(self):

        return self.sim.getObjectVelocity(self.cm_handle)
