#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import torch
import numpy as np
import os
import argparse
from einops import rearrange
import cv2
import h5py

os.environ["HYDRA_FULL_ERROR"] = "1"

from collections import deque

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import threading
import dill
import threading
import hydra
from lightning.pytorch import LightningModule
from pytorch_lightning import seed_everything
from source.common.pytorch_util import dict_apply
from pathlib import Path
from termcolor import cprint

import sys
import numpy as np

sys.path.append("./")


class RosOperator:
    """
    RosOperator
    一个用于多相机、双臂机器人场景的 ROS 辅助类。负责订阅图像、深度、关节状态和里程计话题，
    将最近的消息保存到有界 deque 中，通过 CvBridge 将 ROS Image 转为 OpenCV 图像，并提供
    用于组装同步帧与发布机器人臂/底盘命令的工具方法。

    初始化与参数
    - __init__(args): 初始化内部结构、锁和线程，并调用 init() 与 init_ros() 来建立 deque、锁
      以及 ROS 的订阅/发布器。
    - 期望 args 包含：
        img_left_topic, img_right_topic, img_front_topic
        img_left_depth_topic, img_right_depth_topic, img_front_depth_topic
        puppet_arm_left_topic, puppet_arm_right_topic, robot_base_topic
        puppet_arm_left_cmd_topic, puppet_arm_right_cmd_topic, robot_base_cmd_topic
        use_depth_image (bool), use_robot_base (bool)
        publish_rate (Hz), arm_steps_length (每关节步长列表)

    主要属性（要点）
    - bridge: CvBridge 实例，用于 Image -> OpenCV 转换
    - img_*_deque / img_*_depth_deque: 存放最近 ROS Image 消息的 deque（最多约 2000 帧）
    - puppet_arm_left_deque / puppet_arm_right_deque: 存放 JointState 的 deque
    - robot_base_deque: 存放 Odometry 的 deque
    - puppet_arm_left_publisher / puppet_arm_right_publisher: JointState 命令发布器
    - robot_base_publisher: Twist 命令发布器
    - puppet_arm_publish_thread: 用于连续发布臂命令的后台线程
    - puppet_arm_publish_lock: 协调连续发布线程启停的锁
    - ctrl_state / ctrl_state_lock: 来自控制话题的布尔状态及其互斥保护

    线程与并发说明
    - 在 init() 中会 acquire puppet_arm_publish_lock 表示当前无连续发布线程在运行；
      连续发布线程通过检测该锁以响应停止请求。
    - ctrl_state 使用 ctrl_state_lock 保护以保证线程安全读写。
    - ROS 回调仅向 deque 添加消息，假定 ROS 回调线程与 CPython GIL 的组合足够保证
      在本使用模式下的安全性；若从多个非 ROS 线程访问 deque，建议额外加锁。

    主要方法简介
    - init(): 初始化 CvBridge、各类 deque 以及发布锁（并立即 acquire 以表示“无连续发布线程”）。
    - init_ros(): 初始化 ROS 节点，订阅配置的话题（含深度图像可选），并创建发布器。
    - puppet_arm_publish(left, right): 发布单次 JointState 指令到左右臂（填充 Header 和固定关节名）。
    - robot_base_publish(vel): 发布单次 Twist 指令给底盘（使用 vel[0] 作为线速度 x，vel[1] 作为角速度 z）。
    - puppet_arm_publish_continuous(left, right):
          从 deque 中读取最新观测关节位置，按每关节最大步长逐步逼近目标位置并在每步发布 JointState，
          周期由 args.publish_rate 决定。线程会周期性检查 puppet_arm_publish_lock 以决定是否中止。
    - puppet_arm_publish_linear(left, right):
          使用 np.linspace 在固定步数（默认 100）上生成线性轨迹，以固定频率发布，每步保证最后关节值等于目标。
    - puppet_arm_publish_continuous_thread(left, right):
          管理连续发布线程的替换：如果已有线程则请求其停止、join，然后启动新线程。
    - get_frame():
          尝试从前/左/右相机（及可选深度流）、左右臂关节与可选底盘里程计中组装时间同步的一帧数据。
          同步策略：以所需主题最新消息时间戳的最小值为参考 frame_time，确保每个主题至少
          有一条消息 timestamp >= frame_time；若不能满足返回 None。会丢弃过早的消息并返回与
          frame_time 对应的消息（图像使用 CvBridge.imgmsg_to_cv2(..., 'passthrough') 转换）。
          返回格式：
            (img_front_cv, img_left_cv, img_right_cv,
             img_front_depth_cv 或 None, img_left_depth_cv 或 None, img_right_depth_cv 或 None,
             puppet_arm_left_msg, puppet_arm_right_msg, robot_base_msg 或 None)

    回调方法
    - img_*_callback, img_*_depth_callback, puppet_arm_*_callback, robot_base_callback:
          将收到的消息追加到对应 deque，并在超过最大长度时从左侧弹出旧消息。

    控制状态接口
    - ctrl_callback(msg): 从 Bool/类似消息更新 ctrl_state（线程安全）。
    - get_ctrl_state(): 以线程安全方式返回最新 ctrl_state。

    使用注意
    - 依赖 rospy、sensor_msgs.msg.Image、sensor_msgs.msg.JointState、nav_msgs.msg.Odometry、
      geometry_msgs.msg.Twist、std_msgs.msg.Header、CvBridge、numpy、threading 等库。
    - 若在多个非 ROS 线程中频繁访问 deque，请添加显式锁以保证完整线程安全。
    - 连续发布线程停止/替换依赖 puppet_arm_publish_lock 的正确 acquire/release 协同。
    """
    
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)

        left_arm = None
        right_arm = None

        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return None
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return None
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return None
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return None
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return None
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return None
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return None
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return None
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return None
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return None

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


class EnvRunner:
    """
    EnvRunner
    高级环境运行器与 ROS 接口辅助类，用于收集并堆叠观测，以及向机器人子系统（傀儡臂与底盘）发布动作。
    设计为与提供 ROS IO 方法的 'ros_operator' 对象配合使用（所需接口见下文）。

    参数
    ----------
    n_obs_steps : int, optional
        用于策略输入的历史观测步数（默认：3）。
    n_action_steps : int, optional
        动作输出的预期时域长度（默认：8）。主要用于解释动作形状；类本身不强制数值限制。
    ros_operator : object, optional
        提供 ROS IO 方法的对象。见“所需 ros_operator 接口”。

    属性
    ----------
    n_obs_steps : int
        同参数说明。
    n_action_steps : int
        同参数说明。
    ros_operator : object
        用于发布与接收 ROS 消息的 ros_operator。
    obs : collections.deque
        保存最近观测的 FIFO 缓冲区。deque 长度为 n_obs_steps + 1 以便堆叠最近 n_obs_steps 帧。

    所需的 ros_operator 接口
    -------------------------------
    提供的 ros_operator 至少应实现以下方法：
    - get_frame() -> Any 或 None
        返回最新帧/观测，若不可用则返回 None。
    - puppet_arm_publish(left, right)
        发布单步左右臂命令。
    - robot_base_publish(vel_action)
        （可选）发布底盘速度命令。
    - puppet_arm_publish_continuous(left, right)
        用于臂初始化/复位的连续目标发布。

    行为与主要方法
    -------------------------
    get_frame() -> Any 或 None
        以固定频率轮询 ros_operator.get_frame()，直到获得非 None 的帧或超出重试限制。
        成功返回帧，超时或 ROS 关闭时返回 None。该方法依赖 rospy 的 rate/sleep 与 shutdown 状态。
    step(left_action, right_action, vel_action=None) -> None
        发布单步控制：
        - 始终通过 puppet_arm_publish 发布左右臂动作。
        - 如提供 vel_action，则调用 robot_base_publish 发布底盘速度。
        若 left_action 或 right_action 为 None，则触发断言错误。
    reset() -> None
        通过调用 puppet_arm_publish_continuous 发送一小段预定义臂命令，将机器人置于名义复位姿态，并清空观测缓冲（调用 reset_obs()）。
    stack_last_n_obs(all_obs, n_steps) -> np.ndarray 或 torch.Tensor
        将列表或类似容器中最后 n_steps 个观测堆叠成时间轴在首维的数组/张量：
        - 支持 numpy.ndarray 与 torch.Tensor 两种元素类型。
        - 若可用观测少于 n_steps，则用最早可用帧在前端填充（重复填充）。
        - 不支持的元素类型会抛出 RuntimeError。
    reset_obs() -> None
        清空内部观测缓冲。
    update_obs(current_obs) -> None
        将 current_obs 追加到内部 deque。current_obs 的结构应与策略期望的输入一致（通常为 modality 键映射到 numpy 数组或张量）。
    get_n_steps_obs() -> dict
        返回一个字典：每个键对应一种观测模态，值为由 stack_last_n_obs 返回的最近 n_obs_steps 帧堆叠结果。
        若未记录任何观测则断言失败。
    get_action(policy) -> list
        构建策略兼容的观测字典流程：
        1. 调用 get_n_steps_obs() 获取堆叠的 numpy 数组。
        2. 将每个 numpy 数组转换为 torch.Tensor，并移动到 policy.device 与 policy.dtype。
        3. 在 torch.no_grad() 下调用 policy.predict_action(obs_dict)。
        4. 将结果动作张量转回 CPU numpy。
        5. 提取 'action' 条目，移除批次维度并返回为 Python 列表（按时间步拆分）。
        对 policy 的期望：
        - 属性：policy.device, policy.dtype
        - 方法：policy.predict_action(obs_dict) -> 类 dict，包含键 'action'，其值为形状类似 (batch=1, time, ...) 的张量。

    异常与错误
    ---------------------
    - 当观测缓冲为空时，get_n_steps_obs() 会抛出 AssertionError。
    - 对不支持的观测元素类型，stack_last_n_obs() 会抛出 RuntimeError。
    - step() 会断言 left_action 与 right_action 非 None。

    备注
    -----
    - 假设环境中可用 torch 与 numpy，推理时应使用 torch.no_grad() 以避免梯度计算。
    - get_frame() 中的时序、重试与关闭处理依赖 rospy 与 ROS 环境；在非 ROS 环境下行为可能不同。
    - 调用者负责确保 get_action() 返回的动作对环境有效，以及在调用 step() 前执行任何必要的安全检查。

    示例（概念性）
    --------------------
    # runner = EnvRunner(n_obs_steps=3, ros_operator=my_ros_operator)
    # runner.reset()
    # runner.update_obs(received_obs)
    # action_list = runner.get_action(my_policy)
    # runner.step(action_list[0], action_list[1], vel_action=...)
    """
    
    def __init__(self, n_obs_steps=3, n_action_steps=8, ros_operator=None):
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.ros_operator = ros_operator

        self.obs = deque(maxlen=n_obs_steps+1)

    def get_frame(self):
        cnt = 0
        rate = rospy.Rate(8)
        while True and not rospy.is_shutdown():
            result = self.ros_operator.get_frame()
            if result is not None:
                return result
            
            cnt = cnt + 1
            if cnt > 20:
                return None
            print("syn fail")
            rate.sleep()
        
    def step(self, left_action, right_action, vel_action=None):
        assert left_action is not None, right_action is not None

        self.ros_operator.puppet_arm_publish(left_action, right_action)
        if vel_action is not None:
            self.ros_operator.robot_base_publish(vel_action)
    
    def reset(self):
        left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
        right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
        left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
        right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
        
        self.ros_operator.puppet_arm_publish_continuous(left0, right0)
        self.ros_operator.puppet_arm_publish_continuous(left1, right1)

        self.reset_obs()

    def stack_last_n_obs(self, all_obs, n_steps):
        assert(len(all_obs) > 0)
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
        return result
    
    def reset_obs(self):
        self.obs.clear()

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def get_n_steps_obs(self):
        assert(len(self.obs) > 0), 'no observation is recorded, please update obs first'

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs(
                [obs[key] for obs in self.obs],
                self.n_obs_steps
            )

        return result

    def get_action(self, policy):
        device, dtype = policy.device, policy.dtype
        obs = self.get_n_steps_obs()

        # create obs dict
        np_obs_dict = dict(obs)
        # device transfer
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device).unsqueeze(0))

        # run policy
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)

        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
        action_array = np_action_dict['action'].squeeze(0)
        action_list = [action_array[i] for i in range(action_array.shape[0])]
        # action_list = [action_array[i] for i in range(4)]
        return action_list


def get_model_input(frame):
    (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
            puppet_arm_left, puppet_arm_right, robot_base) = frame
    # img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(f'demo/{img_cnt}.png', img_front)

    img_front = cv2.resize(img_front, (256, 256), interpolation=cv2.INTER_LINEAR)
    # img_left = cv2.resize(img_left, (256, 256), interpolation=cv2.INTER_LINEAR)
    # img_right = cv2.resize(img_right, (256, 256), interpolation=cv2.INTER_LINEAR)

    # img_front = rearrange(img_front, 'h w c -> c h w') / 255.0  
    img_front = rearrange(img_front, 'h w c -> c h w') 
    # img_left = rearrange(img_left, 'h w c -> c h w') / 255.0
    # img_right = rearrange(img_right, 'h w c -> c h w') / 255.0
    qpos = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)

    return dict(
        cam_high = img_front,
        # cam_left = img_left,
        # cam_right = img_right,
        qpos = qpos
    )


def model_inference(args):

    
    # 1. 定义保存路径
    # save_dir  = Path('demo')
    # save_dir.mkdir(parents=True, exist_ok=True)
    # left_dir  = save_dir / 'left'
    # right_dir = save_dir / 'right'

    # left_dir.mkdir(parents=True, exist_ok=True)
    # right_dir.mkdir(parents=True, exist_ok=True)



    # 1 load policy 加载模型
    # treat args.ckpt_dir as the full checkpoint file path
    payload = torch.load(args.ckpt_dir, pickle_module=dill)
    # seed_everything(payload['cfg']['seed'])
    # payload['cfg']["horizon"] = 32
    # payload['cfg']["policy"]["horizon"] = 32
    # payload['cfg']["policy"]["n_action_steps"] = 30
    
    policy: LightningModule = hydra.utils.instantiate(payload['cfg']["policy"])
    # dataset = hydra.utils.instantiate(payload['cfg']["task"]["dataset"])
    # policy.set_normalizer(dataset.get_normalizer())
    
    missing, unexpected = policy.load_state_dict(payload['state_dict'], strict=False)
    if missing:
        print("missing keys:", missing)
        return
    if unexpected:
        print("unexpected keys:", unexpected)
        return

    policy.cuda()
    policy.eval()

    # 2 create env 创建env
    ros_operator = RosOperator(args)
    env = EnvRunner(ros_operator=ros_operator)
    env.reset()

    # 获得输入
    frame = env.get_frame()
    obs = get_model_input(frame)
    env.update_obs(obs)
    
    old_action = None
    
    right1 = np.array([0.8215389, 2.1289692, -1.8642251, -0.28003284, 1.1474806, 0.8877168, 0.0])

    with torch.inference_mode():
        publish_step = 0
        rate = rospy.Rate(args.publish_rate)
        
        gripper_switch = 0
        while True and not rospy.is_shutdown():
            if publish_step % 5 == 0:
                print("1")

            publish_step = publish_step + 1
            if publish_step > args.max_publish_step:
                break
  

            actions = env.get_action(policy) # 获得action
            actions = np.array(actions)
            
            # if gripper_switch == 1:
            #     hack_action = old_action
            #     hack_action[7:14] = right1
            #     actions = np.stack([old_action, hack_action])
            #     actions = action_topp(actions, num=4)
            #     gripper_switch += 1
            #     cprint('right hack','red')
            
            actions = action_topp(actions, num=0) # 插值

                 
            for action in actions:
                if rospy.is_shutdown():
                    break
                
                new_action = np.array(action, copy=True)

                left_action = action[:7] 
                right_action = action[7:14]
                # 处理夹抓状态
                cprint(f'left_gripper:{left_action[-1]}, right_gripper:{right_action[-1]}','yellow')
                left_action[-1] = left_action[-1] if left_action[-1] > 0.025 else 0
                right_action[-1] = right_action[-1] if right_action[-1] > 0.055 else 0
                # if right_action[-1] > 0.5:
                #     right_action[-1] = 0.09
                # elif right_action[-1] < 0.03:
                #     right_action[-1] = 0
                
                vel_action = None
                if args.use_robot_base:
                    vel_action = action[14:16]

                action_diff = action_mse(old_action, new_action)
                left_gripper_diff, right_gripper_diff = gripper_diff(old_action, new_action)
                print(f'Action diff mse: {action_diff}')
                
                if action_diff > 0.05:
                    cprint('Action jump detected, skip this action','red')
                    continue
                # if right_gripper_diff > 0.1 and right_action[-1] > 0.03:
                #     cprint('Right gripper jump detected, skip this action','red')
                #     continue
                # if gripper_switch == 0 and new_action[13] < 0.001:
                #     gripper_switch += 1
                #     cprint('gripper_switch == 1','red')

                env.step(left_action, right_action, vel_action)

                frame = env.get_frame()
                obs = get_model_input(frame)
                env.update_obs(obs)

                rate.sleep()
                
                old_action = new_action
                
def gripper_diff(old_action, new_action):
    if old_action is None or new_action is None:
        # 初始动作不设置保护
        return 0, 0
    
    left_gripper_diff = abs(old_action[6]-new_action[6]) 
    right_gripper_diff = abs(old_action[13]-new_action[13])

    return left_gripper_diff, right_gripper_diff


def action_mse(old_action, new_action):
    if old_action is None or new_action is None:
        # 初始动作不设置保护
        return 0

    def to_array(x):
        # 将输入转换为 numpy 数组
        if isinstance(x, list):
            x = np.stack([np.asarray(v) for v in x])
        else:
            x = np.asarray(x)
        return x.astype(np.float64)

    A = to_array(old_action)
    B = to_array(new_action)

    # exact shape match
    if A.shape == B.shape:
        diff = A - B
    else:
        # 如果首个维度（时间维）不同但其余维度匹配，则在最小时间长度上比较
        if A.ndim > 0 and B.ndim > 0 and A.shape[1:] == B.shape[1:]:
            t = min(A.shape[0], B.shape[0])
            if t == 0:
                return 0.0
            diff = A[:t] - B[:t]
        else:
            a_flat = A.ravel()
            b_flat = B.ravel()
            L = min(a_flat.size, b_flat.size)
            if L == 0:
                return 0.0
            diff = a_flat[:L] - b_flat[:L]

    return float(np.mean(diff ** 2))

def action_topp(actions, num=8):
    """
    对 actions 的每对相邻帧插入 num 个线性插值点。
    - actions: list / np.ndarray / torch.Tensor，时间维在第0维
    - num: 每个相邻对之间插入的中间帧数量（不包含两端）
    返回与输入类型一致（torch.Tensor 会保持原 device 与 dtype，否则返回 np.ndarray）。
    """
    is_torch = torch.is_tensor(actions)
    if is_torch:
        # 如果输入是torch张量，则先把它转换为numpy数组，再转换回来
        device = actions.device
        dtype = actions.dtype
        arr = actions.detach().cpu().numpy()
    else:
        arr = np.asarray(actions)

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if num <= 0 or arr.shape[0] < 2:
        # 如果不需要插值或时间维长度不足2，直接返回原始数据
        out = arr.copy()
        if is_torch:
            return torch.tensor(out, device=device, dtype=dtype)
        return out

    pieces = []
    T = arr.shape[0]
    for i in range(T - 1):
        a = arr[i]
        b = arr[i + 1]
        # 生成包含两端点的 num+2 个点，然后去掉最后一个以避免重复
        segment = np.linspace(a, b, num=num + 2, axis=0)
        pieces.append(segment[:-1])
    pieces.append(arr[-1:].copy())
    out = np.vstack(pieces)

    if is_torch:
        return torch.tensor(out, device=device, dtype=dtype)
    return out
    

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt file path (full path to checkpoint file)', required=True)
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=250, required=False)

    # ROS topic输入和配置，一般不需要修改
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=30, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    model_inference(args)


if __name__ == '__main__':
    main()