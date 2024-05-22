import random
import gym
from gym import spaces
import pybullet as p
import pybullet_data
import math
import numpy as np

from stable_baselines3 import TD3

def newton_iteration(theta,v):
    l = 1.389
    l1 = 0.7
    g = 9.8
    # 设置初值
    x = 0
    # 求解稳态时的运动转向角
    for i in range(20):
        f = math.tan(x) * (math.sqrt((l**2) + (l1**2) * (math.tan(x)**2)) + 0.4407) - ((l / v)**2) * g * math.tan(theta)
        diff_f = (1 + (math.tan(x)**2)) * (math.sqrt((l**2) + (l1**2) * (math.tan(x)**2)) + 0.4407) +\
                 math.tan(x) * ((l1**2) * math.tan(x) * (1 + (math.tan(x)**2))) / (math.sqrt((l**2) + (l1**2) * (math.tan(x)**2)))
        x = x - f / diff_f

    # 由运动转向角求稳态时的车把角
    x = math.atan((math.tan(x) * math.cos(theta)) / (math.cos(0.12222) + math.tan(x) * math.sin(0.12222) * math.sin(theta)))

    return x
def steady_state_calculation(x,v):
    # 输入速度 目标运动转向角 输出对应稳态时刻的车身倾角
    l = 1.489
    l1 = 0.7
    g = 9.8
    # 求解稳态时的运动转向角

    theta = (math.tan(x) * (math.sqrt((l**2) + (l1**2) * (math.tan(x)**2)) + 0.4407))/(((l / v)**2) * g)
    theta = math.atan(theta)
    return theta

class Attitude_control(gym.Env):

    def __init__(self, render: bool = False):
        self._render = render

        # 定义动作空间
        self.action_space = spaces.Box(
            low=np.array([-1.]),
            high=np.array([1.]),
            dtype=np.float32
        )
        # [倾斜角与目标倾斜角度差，倾斜角，倾斜角速度,平面速度]
        # 定义状态空间
        self.observation_space = spaces.Box(
            low=np.array([-1.,-1., -1.,-1.]),
            high=np.array([1.,1.,  1., 1.,]),
            dtype=np.float32
        )

        # 连接引擎
        self._physics_client_id = p.connect(p.GUI if self._render else p.DIRECT)
        # 添加资源路径
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # 设置交互周期
        self.cycle = 1 / 30
        p.setTimeStep(self.cycle)
        # 单个epoch最大交互步
        self.max_step_num = 1000
        # 计算一个epoch的平均角度误差
        self.angle_error = 0
        # 记录每个epoch的平均角度误差
        self.angle_error_csv = []
        # 用于计算目标角度与实际角度的差
        self.target_theta = 0
        # 用于计算倾斜角速度
        self.theta0_old = 0
        # 用于计算车仰角速度
        self.theta1_old = 0
        # 时间步计数器 记录一个epoch的时间步
        self.step_num = 0
        # epoch计数器 记录当前是第几个epoch
        self.epoch_num = 0
        # 记录当前回合累计奖励
        self.r = 0
        # 记录最近20个epoch的平均奖励
        self.epoch_r_list = []
        # 设置环境重力加速度
        p.setGravity(0, 0, -9.8)

        # 记录数据
        self.record_flag = 0
        self.target_theta_csv = []
        self.theta0_csv = []
        self.target_handle_angle_csv = []
        self.handle_angle_csv = []
        self.pre_handle_angel_csv = []

        self.lqr = 0
        self.lqi = 0
        self.rl = 0
        self.rl_lqr = 0

        self.test = 0


        # 加载地形
        p.loadURDF("plane.urdf",globalScaling=2)
        #p.loadURDF(r"3D\terrain.urdf",[-15,-18.5,0],globalScaling=5)
        # 加载自行车，并设置加载的机器人的位姿
        startPos = [0, 0, 1]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        path =r"3D\bike\urdf\bike.urdf"
        #path = r"3D\bicycle.SLDASM\urdf\bicycle.SLDASM.urdf"
        self.bike = p.loadURDF(path, startPos, startOrientation)

        p.changeDynamics(bodyUniqueId=self.bike, linkIndex=2, restitution=0.5, contactStiffness=10 ** 8,
                         contactDamping=10 ** 5)
        p.changeDynamics(bodyUniqueId=self.bike, linkIndex=0, restitution=0.5,contactStiffness=10 ** 8,
                         contactDamping=10 ** 5)
        p.changeDynamics(bodyUniqueId=self.bike, linkIndex=-1, restitution=0.5, contactStiffness=10 ** 8,
                         contactDamping=10 ** 5)
        p.changeDynamics(bodyUniqueId=self.bike, linkIndex=1, restitution=0.5, contactStiffness=10 ** 8,
                         contactDamping=10 ** 5)


        print("连杆信息：")
        for link_index in [-1, 0, 1, 2]:
            link_info = p.getDynamicsInfo(self.bike, link_index)
            print(f"\
                            [0]质量: {link_info[0]}\n\
                            [1]横向摩擦系数(lateral friction): {link_info[1]}\n\
                            [2]主惯性矩: {link_info[2]}\n\
                            [3]惯性坐标系在局部关节坐标系中的位置: {link_info[3]}\n\
                            [4]惯性坐标系在局部关节坐标系中的姿态: {link_info[4]}\n\
                            [5]恢复系数: {link_info[5]}\n\
                            [6]滚动摩擦系数(rolling friction): {link_info[6]}\n\
                            [7]扭转摩擦系数(spinning friction): {link_info[7]}\n\
                            [8]接触阻尼(-1表示不可用): {link_info[8]}\n\
                            [9]接触刚度(-1表示不可用): {link_info[9]}\n\
                            [10]物体属性: 1=刚体，2=多刚体，3=软体: {link_info[10]}\n\
                            [11]碰撞边界: {link_info[11]}\n\n")

    def __get_observation(self,target_theta = 0,recoder = 0):
        _, cubeOrn = p.getBasePositionAndOrientation(self.bike)
        t0 = p.getEulerFromQuaternion(cubeOrn)
        # 倾斜角
        theta0 = t0[0]

        # 目标倾斜角与当前倾斜角差
        dis_angle = target_theta - theta0

        # 倾斜角速度
        w0 = (theta0 - self.theta0_old) / self.cycle
        # 车身仰角
        theta1 = t0[1]



        if recoder == 1:
            self.theta0_old = theta0
            self.theta1_old = theta1

        # 平面速度
        vx = p.getBaseVelocity(self.bike)[0][0]
        vy = p.getBaseVelocity(self.bike)[0][1]
        v = math.sqrt(vx**2 + vy**2)

        if recoder:
            self.angle_error += abs(dis_angle)
            self.theta0_csv.append(theta0)
            self.target_theta_csv.append(target_theta)
            self.handle_angle_csv.append(p.getJointState(bodyUniqueId = self.bike, jointIndex = 1)[0])

        # 状态归一化
        dis_angle = np.clip(dis_angle, -1.57, 1.57) / 1.57
        theta0 = np.clip(theta0, -1.57, 1.57) / 1.57
        w0 = np.clip(w0, -10., 10.) / 10
        v = np.clip(v, -5., 5.) / 5


        return np.array([dis_angle,theta0,w0,v],dtype=np.float32)

    def __observation_reduction(self,state):
        dis_angle = state[0] * 1.57
        theta0 = state[1] * 1.57
        w0 = state[2] * 10
        v = state[3] * 5
        return np.array([dis_angle,theta0,w0,v])

    def __calculate_reward(self,state_last,state):
        state_last = self.__observation_reduction(state_last)
        state = self.__observation_reduction(state)
        dis_angle = state_last[0]
        w0 = state[2]


        beta = 0.002
        if abs(dis_angle) < beta:
            reward = 0.1 - abs(w0)
        else:
            reward = abs(state_last[0]) - abs(state[0])
        return reward

    def reset(self):
        self.epoch_r_list.append(self.r)
        print("Episode:"+str(self.epoch_num) + "   time step："+str(self.step_num) + "   Reward:"+str(self.r)
              +"   Mean_reawrd:"+str(np.mean(self.epoch_r_list[-20:]))+"    angle_error:"+str(self.angle_error/(1+self.step_num)))

        self.target_theta = 0
        self.step_num = 0
        self.r = 0
        self.epoch_num += 1
        # 用于计算倾斜角速度
        self.theta0_old = 0
        # 用于计算车仰角速度
        self.theta1_old = 0
        self.angle_error = 0

        self.target_theta_csv = []
        self.theta0_csv = []
        self.handle_angle_csv = []
        self.target_handle_angle_csv = []
        self.pre_handle_angel_csv = []

        self.error_sum = 0

        startPos = [0,0,0.75]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.bike, startPos, startOrientation)
        p.resetJointState(self.bike, 0, 0, 0)
        p.resetJointState(self.bike, 1, 0, 0)
        p.resetJointState(self.bike, 2, 0, 0)
        p.stepSimulation()

        return self.__get_observation(self.target_theta)

    def step(self, action):
        location, _ = p.getBasePositionAndOrientation(self.bike)
        p.resetDebugVisualizerCamera(
            cameraDistance=2,
            cameraYaw=-70,
            cameraPitch=-10.0001,
            cameraTargetPosition=location
        )

        state_last =self.__get_observation(self.target_theta,recoder=1)
        self.target_handle_angle_csv.append(action[0] * math.pi/4)
        if (self.step_num % 100 == 0):
            sigma =min(0.5,(1.2**(self.epoch_num))*0.1)
            self.target_theta = np.random.normal(loc=0.0, scale=sigma, size=None)

        if (self.target_theta >= math.pi/12) or (self.target_theta <= -math.pi/12):
            self.target_theta = random.uniform(-math.pi/12,math.pi/12)

        if self.test:
            if (self.step_num <= 100):
                self.target_theta = 0
            elif(self.step_num > 100) and (self.step_num <= 200):
                self.target_theta = -0.13
            elif(self.step_num > 200) and (self.step_num <= 300):
                self.target_theta = 0.21
            else:
                self.target_theta = 0.07


        ########################################################################
        err = self.target_theta - ((state_last[1])*1.57)
        self.error_sum += err/30

        # 状态反馈+前馈
        if self.step_num < 100:
            kp = 15.2491
            kd = 2.96
            k = 0
            self.target_theta = 0
        else:
            kp = 15.2491
            kd = 2.96
            k = 12.3

        if self.lqr or self.rl_lqr:
            control = kp * state_last[1]*1.57 + kd * state_last[2]*10 - self.target_theta * k   #lqr
        elif self.lqi:
            control = kp * state_last[1] * 1.57 + kd * state_last[2] * 10 - self.error_sum * 15   #lqi
        elif self.rl:
            control = 0

        control = math.atan((control * 2.25 / (4**2)))

        if (control > 0.785):
            control = 0.785
        if (control < -0.785):
            control = -0.785
        ########################################################################
        # 后轮
        p.setJointMotorControl2(
            bodyUniqueId=self.bike,
            jointIndex=0,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-10,
            force=20
        )
        # 前轮
        p.setJointMotorControl2(
            bodyUniqueId=self.bike,
            jointIndex=2,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-10,
            force=20
        )

        if self.lqr or self.lqi:
            u = control
        elif self.rl:
            u = action[0] * math.pi/4
        elif self.rl_lqr:
            u = action[0] * 0.15 + control

        # 车把角
        p.setJointMotorControl2(
            bodyUniqueId=self.bike,
            jointIndex=1,
            controlMode=p.POSITION_CONTROL,
            targetPosition=u-0.0348,   #车把矫正
            force=200,
            positionGain=0.3
        )

        p.stepSimulation()
        state = self.__get_observation(self.target_theta)
        reward = self.__calculate_reward(state_last,state)
        # 判断回合是否结束
        if ((self.step_num) >= self.max_step_num):
            done = True
        elif(abs(1.57*state[1]) > (math.pi/3)):
            done = True
            reward = reward - 5
        else:
            done = False

        if done:
            data_processing_matrix = np.array([[0.], [0], [0], [0.0348]])
            self.angle_error_csv.append(self.angle_error/self.step_num)
            np.savetxt('model\dangele_error.csv', np.array([self.angle_error_csv]).T, delimiter=',')
            if (self.record_flag == 1):
                np.savetxt('model\data.csv', (np.array([self.target_theta_csv[:-1],
                                                       self.theta0_csv[:-1],
                                                       self.target_handle_angle_csv[:-1],
                                                       self.handle_angle_csv[:-1]
                ])+data_processing_matrix).T,delimiter=',')

        info = {}
        self.step_num += 1
        self.r += reward
        return state, reward, done, info


    def seed(self, seed=None):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect()
        self._physics_client_id = -1

class Position_control_td3_complex_line(gym.Env):

    def __init__(self, render: bool = False):
        self._render = render

        # 定义动作空间()
        self.action_space = spaces.Box(
            low=np.array([-1.]),
            high=np.array([1.]),
            dtype=np.float32
        )
        # [横向误差,航向误差,后轮线速度]
        # 定义状态空间
        self.observation_space = spaces.Box(
            low=np.array([-1.,-1., -1.,-1.,-1.,-1.]),
            high=np.array([1.,1.,  1., 1., 1.,1.]),
            dtype=np.float32
        )

        # 连接引擎
        self._physics_client_id = p.connect(p.GUI if self._render else p.DIRECT)
        # 添加资源路径
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # 设置交互周期
        self.cycle = 1 / 30
        p.setTimeStep(self.cycle)
        # 单个epoch最大交互步
        self.max_step_num = 1500

        # 用于计算目标角度与实际角度的差
        self.target_theta = 0
        # 用于计算倾斜角速度
        self.theta0_old = 0
        # 用于计算车仰角速度
        self.theta1_old = 0
        # 时间步计数器 记录一个epoch的时间步
        self.step_num = 0
        # epoch计数器 记录当前是第几个epoch
        self.epoch_num = 0
        # 记录当前回合累计奖励
        self.r = 0
        # 记录最近20个epoch的平均奖励
        self.epoch_r_list = []
        # 设置环境重力加速度
        p.setGravity(0, 0, -9.8)
        # 记录数据
        self.record_flag = 0
        self.target_theta_csv = []
        self.theta0_csv = []
        self.target_handle_angle_csv = []
        self.handle_angle_csv = []

        self.later_error_csv = []
        self.course_error_angle_csv = []
        self.X = []
        self.Y = []

        self.position_error_csv = []
        self.compensate = 0

        self.stanley = 0
        self.rl = 0
        self.rl_stanley = 0

        self.last_w0 = 0



        self.attitude_control_model = TD3.load("model/attitude_rl_lqr.zip")

        # 创建视觉形状模型
        # 创建视觉形状模型
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName="3D\complex_path.obj",
            rgbaColor=[255 / 255, 50 / 255, 0 / 255, 0.5],
            specularColor=[0.4, 0.4, 0.8],
            visualFramePosition=[-1, -1, 0],
            #visualFrameOrientation=p.getQuaternionFromEuler([0, -math.pi/2, -math.pi/2]),
            meshScale=[0.001, 0.001, 0.001]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id,
        )

        # 加载地形
        p.loadURDF("plane.urdf",globalScaling=2)
        #p.loadURDF(r"3D\terrain.urdf",[-15,-18.5,0],globalScaling=5)
        # 加载自行车，并设置加载的机器人的位姿
        startPos = [0, 0, 1]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        path = r"3D\bike\urdf\bike.urdf"
        self.bike = p.loadURDF(path, startPos, startOrientation)

        p.changeDynamics(bodyUniqueId=self.bike, linkIndex=2, restitution=0.5, contactStiffness=10 ** 8,
                         contactDamping=10 ** 5)
        p.changeDynamics(bodyUniqueId=self.bike, linkIndex=0, restitution=0.5,contactStiffness=10 ** 8,
                         contactDamping=10 ** 5)
        p.changeDynamics(bodyUniqueId=self.bike, linkIndex=-1, restitution=0.5, contactStiffness=10 ** 8,
                         contactDamping=10 ** 5)
        p.changeDynamics(bodyUniqueId=self.bike, linkIndex=1, restitution=0.5, contactStiffness=10 ** 8,
                         contactDamping=10 ** 5)


        print("连杆信息：")
        for link_index in [-1, 0, 1, 2]:
            link_info = p.getDynamicsInfo(self.bike, link_index)
            print(f"\
                            [0]质量: {link_info[0]}\n\
                            [1]横向摩擦系数(lateral friction): {link_info[1]}\n\
                            [2]主惯性矩: {link_info[2]}\n\
                            [3]惯性坐标系在局部关节坐标系中的位置: {link_info[3]}\n\
                            [4]惯性坐标系在局部关节坐标系中的姿态: {link_info[4]}\n\
                            [5]恢复系数: {link_info[5]}\n\
                            [6]滚动摩擦系数(rolling friction): {link_info[6]}\n\
                            [7]扭转摩擦系数(spinning friction): {link_info[7]}\n\
                            [8]接触阻尼(-1表示不可用): {link_info[8]}\n\
                            [9]接触刚度(-1表示不可用): {link_info[9]}\n\
                            [10]物体属性: 1=刚体，2=多刚体，3=软体: {link_info[10]}\n\
                            [11]碰撞边界: {link_info[11]}\n\n")
    def get_state_attitude_control(self,target_theta = 0,recoder = 0):
        _, cubeOrn = p.getBasePositionAndOrientation(self.bike)
        t0 = p.getEulerFromQuaternion(cubeOrn)
        # 倾斜角
        theta0 = t0[0]

        # 目标倾斜角与当前倾斜角差
        dis_angle = target_theta - theta0

        # 倾斜角速度
        w0 = (theta0 - self.theta0_old) / self.cycle


        if recoder == 1:
            self.theta0_old = theta0


        # 平面速度
        vx = p.getBaseVelocity(self.bike)[0][0]
        vy = p.getBaseVelocity(self.bike)[0][1]
        v = math.sqrt(vx ** 2 + vy ** 2)


        if recoder:
            if self.rl_stanley:
                self.theta0_csv.append(theta0 + self.compensate)
                self.target_theta_csv.append(target_theta + self.compensate)
            else:
                self.theta0_csv.append(theta0 - self.compensate)
                self.target_theta_csv.append(target_theta - self.compensate)


            self.handle_angle_csv.append(p.getJointState(bodyUniqueId=self.bike, jointIndex=1)[0])

        # 状态归一化
        dis_angle = np.clip(dis_angle, -1.57, 1.57) / 1.57
        theta0 = np.clip(theta0, -1.57, 1.57) / 1.57
        w0 = np.clip(w0, -10., 10.) / 10
        v = np.clip(v, -5., 5.) / 5



        return np.array([dis_angle, theta0, w0, v], dtype=np.float32)

    def __get_observation(self,recoder = 0):
        _, cubeOrn = p.getBasePositionAndOrientation(self.bike)
        t0 = p.getEulerFromQuaternion(cubeOrn)

        # 倾斜角
        theta0 = t0[0]

        # 倾斜角速度
        w = (theta0 - self.theta0_old) / self.cycle

        w0 = w * 0.8 + 0.2 * self.last_w0

        self.last_w0 = w0



        # 平面速度
        vx = p.getBaseVelocity(self.bike)[0][0]
        vy = p.getBaseVelocity(self.bike)[0][1]
        v = math.sqrt(vx ** 2 + vy ** 2)

        # 航向误差角 & 横向误差
        back_wheel_point_x, back_wheel_point_y, _ = p.getLinkState(self.bike, 0)[0]
        forward_wheel_point_x, forward_wheel_point_y, _ = p.getLinkState(self.bike, 2)[0]


        bike_direction_vector = [forward_wheel_point_x - back_wheel_point_x,
                                 forward_wheel_point_y - back_wheel_point_y]
        if (forward_wheel_point_x < 55) and (forward_wheel_point_y < 5):
            modulus = math.sqrt(bike_direction_vector[0] ** 2 + bike_direction_vector[1] ** 2)
            course_error_angle = math.acos(bike_direction_vector[0] / modulus)
            if bike_direction_vector[1] < 0:
                course_error_angle = -course_error_angle

            lateral_error = forward_wheel_point_y
            self.compensate = 0.0188

            k = 0

        elif (forward_wheel_point_x > 55) and (forward_wheel_point_x < 75) and (forward_wheel_point_y > -5) and (forward_wheel_point_y < 15):
            auxiliary_vector = [55 - forward_wheel_point_x, 15 - forward_wheel_point_y]
            modulus_bike = math.sqrt(bike_direction_vector[0] ** 2 + bike_direction_vector[1] ** 2)
            modulus_auxiliary = math.sqrt(auxiliary_vector[0] ** 2 + auxiliary_vector[1] ** 2)
            var = bike_direction_vector[0] * auxiliary_vector[0] + bike_direction_vector[1] * auxiliary_vector[1]
            course_error_angle = math.acos(var / (modulus_bike * modulus_auxiliary))
            course_error_angle = math.pi / 2 - course_error_angle

            lateral_error = 15 - modulus_auxiliary
            self.compensate = 0
            k = 1/15

        elif (forward_wheel_point_x > 65) and (forward_wheel_point_x < 75) and (forward_wheel_point_y > 15) and (forward_wheel_point_y < 35):
            modulus = math.sqrt(bike_direction_vector[0] ** 2 + bike_direction_vector[1] ** 2)
            course_error_angle = math.acos(bike_direction_vector[1] / modulus)
            if bike_direction_vector[0] > 0:
                course_error_angle = -course_error_angle

            lateral_error = -(forward_wheel_point_x - 70)
            self.compensate = 0.0188
            k = 0

        elif (forward_wheel_point_x > 35) and (forward_wheel_point_x < 75) and (forward_wheel_point_y > 35) and (forward_wheel_point_y < 55):
            auxiliary_vector = [55 - forward_wheel_point_x, 35 - forward_wheel_point_y]
            modulus_bike = math.sqrt(bike_direction_vector[0] ** 2 + bike_direction_vector[1] ** 2)
            modulus_auxiliary = math.sqrt(auxiliary_vector[0] ** 2 + auxiliary_vector[1] ** 2)
            var = bike_direction_vector[0] * auxiliary_vector[0] + bike_direction_vector[1] * auxiliary_vector[1]
            course_error_angle = math.acos(var / (modulus_bike * modulus_auxiliary))
            course_error_angle = math.pi / 2 - course_error_angle

            lateral_error = 15 - modulus_auxiliary
            self.compensate = 0
            k = 1/15

        elif (forward_wheel_point_x > 28) and (forward_wheel_point_x < 45) and (forward_wheel_point_y > 18) and (forward_wheel_point_y < 35):
            auxiliary_vector = [28 - forward_wheel_point_x, 35 - forward_wheel_point_y]
            modulus_bike = math.sqrt(bike_direction_vector[0] ** 2 + bike_direction_vector[1] ** 2)
            modulus_auxiliary = math.sqrt(auxiliary_vector[0] ** 2 + auxiliary_vector[1] ** 2)
            var = bike_direction_vector[0] * auxiliary_vector[0] + bike_direction_vector[1] * auxiliary_vector[1]
            course_error_angle = math.acos(var / (modulus_bike * modulus_auxiliary))
            course_error_angle = course_error_angle - math.pi / 2

            lateral_error = modulus_auxiliary - 12
            self.compensate = 0
            k = -1/12


        elif (forward_wheel_point_x > -10) and (forward_wheel_point_x < 28) and (forward_wheel_point_y > 15) and (forward_wheel_point_y < 58):
            modulus = math.sqrt(bike_direction_vector[0] ** 2 + bike_direction_vector[1] ** 2)
            course_error_angle = math.acos(-bike_direction_vector[0] / modulus)
            if bike_direction_vector[1] > 0:
                course_error_angle = -course_error_angle

            lateral_error = 23 - forward_wheel_point_y

            k = 1/12
            self.compensate = -0.0188

        else:
            course_error_angle = 0
            lateral_error = 0
            k = 0


        if recoder :
            self.later_error_csv.append(lateral_error)
            self.course_error_angle_csv.append(course_error_angle)
            self.X.append(forward_wheel_point_x)
            self.Y.append(forward_wheel_point_y)


        # 状态归一化
        lateral_error = np.clip(lateral_error,-10.,10.) / 10
        course_error_angle = np.clip(course_error_angle,-1.57, 1.57) / 1.57
        theta0 = np.clip(theta0, -1.57, 1.57) / 1.57
        w0 = np.clip(w0, -10., 10.) / 10
        v = np.clip(v, -5., 5.) / 5
        k = k * 8

        return np.array([lateral_error,course_error_angle,v,theta0,w0,k],dtype=np.float32)

    def __observation_reduction(self,state):
        lateral_error = state[0] * 10
        course_error_angle = state[1] * 1.57
        v = state[2] * 5
        theta0 = state[3] * 1.57
        w0 = state[4] * 10
        k = state[5] / 8

        return np.array([lateral_error,course_error_angle,v,theta0,w0,k])


    def __calculate_reward(self,state_last,state):
        state_last = self.__observation_reduction(state_last)
        state = self.__observation_reduction(state)

        if abs(state[0]) < 0.05:
            reward = 0.2 - abs(state[1]) - 0.2 * abs(state[4])
        else:
            reward = abs(state_last[0]) - abs(state[0]) + 0.2*(abs(state_last[1]) - abs(state[1]))

        return reward


    def reset(self):
        self.epoch_r_list.append(self.r)
        print("Episode:"+str(self.epoch_num) + "   time step："+str(self.step_num) + "   Reward:"+str(self.r)
              +"   Mean_reawrd:"+str(np.mean(self.epoch_r_list[-20:])) + "     later_error:" + str(sum(map(abs,self.later_error_csv))/(len(self.later_error_csv)+0.00001)))

        self.target_theta = 0
        self.step_num = 0
        self.r = 0
        self.epoch_num += 1
        # 用于计算倾斜角速度
        self.theta0_old = 0
        # 用于计算车仰角速度
        self.theta1_old = 0

        self.target_theta_csv = []
        self.theta0_csv = []
        self.handle_angle_csv = []
        self.target_handle_angle_csv = []

        self.later_error_csv = []
        self.course_error_angle_csv = []
        self.X = []
        self.Y = []

        self.last_w0 = 0


        startPos = [0,0,0.78]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.bike, startPos, startOrientation)
        p.resetJointState(self.bike, 0, 0, 0)
        p.resetJointState(self.bike, 1, 0, 0)
        p.resetJointState(self.bike, 2, 0, 0)
        p.stepSimulation()
        return self.__get_observation(self.target_theta)

    def step(self, action):
        location, _ = p.getBasePositionAndOrientation(self.bike)
        p.resetDebugVisualizerCamera(
            cameraDistance=10,
            cameraYaw=-0,
            cameraPitch=-89.9,
            cameraTargetPosition=location
        )

        state_last =self.__get_observation(recoder=1)

        tar_attitude_rl = action[0] * math.pi/9   # rl补偿值

        # stanley计算目标横滚角
        lateral_error = state_last[0] * 10

        course_error_angle = state_last[1] * 1.57
        x = math.atan(0.6 * lateral_error / (state_last[2] * 5)) + course_error_angle * 0.4
        tar_attitude_stanley = steady_state_calculation(x, state_last[2] * 5)  # 模型转换



        if self.stanley:
            tar_attitude = tar_attitude_stanley      # only stanley
        elif self.rl:
            tar_attitude = tar_attitude_rl  # only rl
        else:
            tar_attitude = tar_attitude_stanley + 0.3 * tar_attitude_rl     # stanley + rl

        if tar_attitude > math.pi / 6:
            tar_attitude = math.pi / 6
        elif tar_attitude < -math.pi / 6:
            tar_attitude = -math.pi / 6


        # res rl 车身倾角控制器
        attitude_state = self.get_state_attitude_control(tar_attitude+self.compensate,recoder=1)
        action0 = self.attitude_control_model.predict(attitude_state)
        kp = 15.249
        kd = 2.96
        k = 12.3
        pid_control = kp * attitude_state[1]*1.57 + kd * attitude_state[2]*10 - tar_attitude * k
        pid_control = math.atan((pid_control * 2.25 / ((4)**2)))

        # 车把角约束
        if (pid_control > 0.785):
            pid_control = 0.785
        if (pid_control < -0.785):
            pid_control = -0.785

        # 后轮
        p.setJointMotorControl2(
            bodyUniqueId=self.bike,
            jointIndex=0,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-10,
            force=20
        )
        # 前轮
        p.setJointMotorControl2(
            bodyUniqueId=self.bike,
            jointIndex=2,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-10,
            force=20
        )
        # 车把角
        p.setJointMotorControl2(
            bodyUniqueId=self.bike,
            jointIndex=1,
            controlMode=p.POSITION_CONTROL,
            targetPosition=action0[0] * 0.15 + pid_control - 0.0348,
            force=200,
            positionGain=0.3
        )
        p.stepSimulation()
        state = self.__get_observation()
        reward = self.__calculate_reward(state_last,state)
        # 判断回合是否结束
        if (self.step_num >= self.max_step_num):
            done = True
        elif(abs(10 * state[0]) >= 3):
            reward = reward - 10
            done = True
        elif(abs(attitude_state[1])*1.57 > math.pi/3):
            reward = reward - 10
            done = True
        elif (abs(state[1]) * 1.57 > math.pi / 3):
            reward = reward - 10
            done = True
        else:
            done = False

        if done:
            data_processing_matrix = np.array([[0], [0],[0.0348], [0.], [0.],[0.],[0.]])
            self.position_error_csv.append(sum(map(abs, self.later_error_csv))/ len(self.later_error_csv))
            np.savetxt('model\position_error.csv', np.array([self.position_error_csv]).T, delimiter=',')
            if (self.record_flag == 1):
                np.savetxt('model\data.csv', np.array([self.target_theta_csv[:-1],
                                                       self.theta0_csv[:-1],
                                                       self.handle_angle_csv[:-1],
                                                       self.later_error_csv[:-1],
                                                       self.course_error_angle_csv[:-1],
                                                       self.X[:-1],
                                                       self.Y[:-1]
                ] + data_processing_matrix).T,delimiter=',')

        info = {}
        self.step_num += 1
        self.r += reward
        return state, reward, done, info


    def seed(self, seed=None):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect()
        self._physics_client_id = -1

