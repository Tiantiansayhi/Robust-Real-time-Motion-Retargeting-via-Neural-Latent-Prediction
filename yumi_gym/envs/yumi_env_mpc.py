import os
import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
import pybullet_data
import numpy as np
import cvxpy
import h5py
import time

#控制参数
MAX_VEL = 3.14159265359  # maximum accel [rad/s]
window = 5
rate = 15/240
t = 1
N=100 # 迭代范围
EPS = 1e-4 # 迭代精度
R = np.diag([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])*0.1
Q = np.diag([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])*25
A = np.diag([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
B = np.diag([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])*rate


def get_nparray_from_matrix(x):
    return np.array(x).flatten()

class YumiEnv(gym.Env):
    def __init__(self):
        super(YumiEnv, self).__init__()
        p.connect(p.DIRECT)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=-20, cameraTargetPosition=[0,0,0.1])
        self.step_counter = 0
        self.joints = ['yumi_joint_1_l',
            'yumi_joint_2_l',
            'yumi_joint_7_l',
            'yumi_joint_3_l',
            'yumi_joint_4_l',
            'yumi_joint_5_l',
            'yumi_joint_6_l',
            'yumi_joint_1_r',
            'yumi_joint_2_r',
            'yumi_joint_7_r',
            'yumi_joint_3_r',
            'yumi_joint_4_r',
            'yumi_joint_5_r',
            'yumi_joint_6_r',
            "link1",
            "link11",
            "link2",
            "link22",
            "link3",
            "link33",
            "link4",
            "link44",
            "link5",
            "link51",
            "link52",
            "link53",
            "Link1",
            "Link11",
            "Link2",
            "Link22",
            "Link3",
            "Link33",
            "Link4",
            "Link44",
            "Link5",
            "Link51",
            "Link52",
            "Link53",
            ]
        self.action_space = spaces.Box(np.array([-1]*len(self.joints)*6), np.array([1]*len(self.joints)*6))
        self.observation_space = spaces.Box(np.array([-1]*len(self.joints)), np.array([1]*len(self.joints)))

    def step(self, action, custom_reward=None):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)         
        
        # get joint states
        jointStates = {}
        for joint in self.joints:
            jointStates[joint] = p.getJointState(self.yumiUid, self.joint2Index[joint]) + p.getLinkState(self.yumiUid, self.joint2Index[joint])

        joint_ang = []
        for value in jointStates.values():
            joint_ang.append(value[0])
            # self.joint.append(value[0])
            # self.v.append(value[1])

              
        # PD control
        # q_error = np.array(action[:38]) - np.array(joint_ang)
        # v_error = np.array(action[38:]) - np.array(action[:38])
        # v_error = np.array(action[38:]) - np.array(joint_ang)
        # tau = self.K*q_error + self.D*v_error
        u = MPC(joint_ang, action)
        # u = LQR(joint_ang, action)
        p.setJointMotorControlArray(self.yumiUid,
                                    [self.joint2Index[joint] for joint in self.joints],
                                    # p.POSITION_CONTROL,
                                    p.VELOCITY_CONTROL,
                                    # targetPositions = action[:38],
                                    targetVelocities = u)
        
        p.stepSimulation() 
        
        # recover color
        for joint, index in self.joint2Index.items():
            if joint in self.jointColor and joint != 'world_joint':
                p.changeVisualShape(self.yumiUid, index, rgbaColor=self.jointColor[joint])

        
        # check collision
        collision = False
        for joint in self.joints:
            if len(p.getContactPoints(bodyA=self.yumiUid, linkIndexA=self.joint2Index[joint])) > 0:
                collision = True
                for contact in p.getContactPoints(bodyA=self.yumiUid, linkIndexA=self.joint2Index[joint]):
                    print("Collision Occurred in Joint {} & Joint {}!!!".format(contact[3], contact[4]))
                    p.changeVisualShape(self.yumiUid, contact[3], rgbaColor=[1, 0, 0, 1])
                    p.changeVisualShape(self.yumiUid, contact[4], rgbaColor=[1, 0, 0, 1])
        
        self.step_counter += 1

        if custom_reward is None:
            # default reward
            reward = 0
            done = False
        else:
            # custom reward
            reward, done = custom_reward(jointStates=jointStates, collision=collision, step_counter=self.step_counter)

        info = {'collision': collision}
        angle = [jointStates[joint][0] for joint in self.joints]
        velocity = [jointStates[joint][1] for joint in self.joints]
        observation = angle + velocity
        return observation, reward, done, info

    def reset(self):
        p.resetSimulation()
        self.step_counter = 0
        self.yumiUid = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/yumi_with_hands.urdf"),
                                  useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        p.setGravity(0, 0, -10)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(1./240.)
        p.setRealTimeSimulation(1)
        self.joint2Index = {} # jointIndex map to jointName
        for i in range(p.getNumJoints(self.yumiUid)):
            self.joint2Index[p.getJointInfo(self.yumiUid, i)[1].decode('utf-8')] = i

        self.jointColor = {} # jointName map to jointColor
        for data in p.getVisualShapeData(self.yumiUid):
            self.jointColor[p.getJointInfo(self.yumiUid, data[1])[1].decode('utf-8')] = data[7]

        # recover color
        for joint, index in self.joint2Index.items():
            if joint in self.jointColor and joint != 'world_joint':
                p.changeVisualShape(self.yumiUid, index, rgbaColor=self.jointColor[joint])

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0, 0.5],
                                                          distance=.7,
                                                          yaw=90,
                                                          pitch=0,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960)/720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        p.disconnect()


def get_nparray_from_matrix(x):
    return np.array(x).flatten()


def MPC(state, goal):
    # end_encoder_time = time.time()
    x_ref = np.array(goal).reshape(-1, 38)  #[5, 38]
    x0 = np.array(state)
    
    x = cvxpy.Variable((window+1, 38))
    u = cvxpy.Variable((window, 38))
    

    cost = 0.0
    constraints = []

    for t in range(window):
        if t != 0:      
            cost += cvxpy.quad_form(x[t,:] - x_ref[t-1,:], Q)

        constraints += [x[t+1, :] == A @ x[t, :] + B @ u[t, :]]        
        cost += cvxpy.quad_form(u[t, :], R)
    
    cost += cvxpy.quad_form(x[window, :] - x_ref[window-1], Q)

    constraints += [(x[0, :]) == x0]
    constraints += [cvxpy.abs(u[:, 0:4]) <= 3.14]
    constraints += [cvxpy.abs(u[:, 7:11]) <= 3.14]
    constraints += [cvxpy.abs(u[:, 4:7]) <= 6.97]
    constraints += [cvxpy.abs(u[:, 11:14]) <= 6.97]
    constraints += [cvxpy.abs(u[:, 14:22]) <= 4.53]
    constraints += [cvxpy.abs(u[:, 26:34]) <= 4.53]
    constraints += [cvxpy.abs(u[:, 22:26]) <= 1.22]
    constraints += [cvxpy.abs(u[:, 34:]) <= 1.22]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver = cvxpy.ECOS, verbose = False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        u = get_nparray_from_matrix(u.value)
    else:
        print("Error:Cannot slove")
        u = None
    # u_list = u[:38].tolist()
    u_list = u[:38].tolist()
    # end_predict_time = time.time()  
    # print('calculate time {} ms'.format((end_predict_time - end_encoder_time)*1000))
    return u_list

def cal_Ricatti(A,B,Q,R):
    """解代数里卡提方程

    Args:
        A (_type_): 状态矩阵A
        B (_type_): 状态矩阵B
        Q (_type_): Q为半正定的状态加权矩阵, 通常取为对角阵；Q矩阵元素变大意味着希望跟踪偏差能够快速趋近于零；
        R (_type_): R为正定的控制加权矩阵，R矩阵元素变大意味着希望控制输入能够尽可能小。

    Returns:
        _type_: _description_
    """
    # 设置迭代初始值
    Qf=Q
    P=Qf
    # 循环迭代
    for t in range(N):
        P_ = Q + A.T @ P @ A-A.T @ P @ B @ np.linalg.pinv( R + B.T @ P @ B) @ B.T @ P @ A
        if(abs(P_-P).max()<EPS):
            break
        P=P_
    return P_

def LQR(state, goal):
    # end_encoder_time = time.time()

    x_ref = np.array(goal[38:]).reshape(-1, 38)  #[5, 38]
    x0 = np.array(state)
    
    x = x0 - x_ref[-1]
    P = cal_Ricatti(A,B,Q,R)

    K = -np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
    u = K @ x

    u_list = u.tolist()
    # end_predict_time = time.time()  
    # print('calculate time {} ms'.format((end_predict_time - end_encoder_time)*1000))
    return u_list


if __name__ == '__main__':
    pre = h5py.File('/home/wtt/Data/online_result/pre/stgcn/action/inference/yumi_intro.h5', 'r') 
    key = '大学-daxue'
    l_glove_angle_pre = pre[key + '/l_hand'][:]   #[362, 6, 12]
    r_glove_angle_pre = pre[key + '/r_hand'][:]  

    l_joint_angle_pre = pre[key + '/l_arm'][:]   #[362, 6, 7]
    r_joint_angle_pre = pre[key + '/r_arm'][:]
    total_frames = l_joint_angle_pre.shape[0]
    
    state = [0]*38
    t = 0
    action = l_joint_angle_pre[t, 0, :].tolist() + r_joint_angle_pre[t, 0, :].tolist() + l_glove_angle_pre[t, 0, :].tolist() + r_glove_angle_pre[t, 0, :].tolist()+\
            l_joint_angle_pre[t, 1, :].tolist() + r_joint_angle_pre[t, 1, :].tolist() + l_glove_angle_pre[t, 1, :].tolist() + r_glove_angle_pre[t, 1, :].tolist()+\
            l_joint_angle_pre[t, 2, :].tolist() + r_joint_angle_pre[t, 2, :].tolist() + l_glove_angle_pre[t, 2, :].tolist() + r_glove_angle_pre[t, 2, :].tolist()+\
            l_joint_angle_pre[t, 3, :].tolist() + r_joint_angle_pre[t, 3, :].tolist() + l_glove_angle_pre[t, 3, :].tolist() + r_glove_angle_pre[t, 3, :].tolist()+\
            l_joint_angle_pre[t, 4, :].tolist() + r_joint_angle_pre[t, 4, :].tolist() + l_glove_angle_pre[t, 4, :].tolist() + r_glove_angle_pre[t, 4, :].tolist()+\
            l_joint_angle_pre[t, 5, :].tolist() + r_joint_angle_pre[t, 5, :].tolist() + l_glove_angle_pre[t, 5, :].tolist() + r_glove_angle_pre[t, 5, :].tolist()

    # u = MPC(state, action)
    u = LQR(state, action)
    print(u)
