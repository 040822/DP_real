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
import time

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

import sys
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


def get_model_input(frame, img_cnt):
    (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
            puppet_arm_left, puppet_arm_right, robot_base) = frame
    # img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(f'demo/{img_cnt}.png', img_front)

    img_front = cv2.resize(img_front, (256, 256), interpolation=cv2.INTER_LINEAR)
    # img_left = cv2.resize(img_left, (256, 256), interpolation=cv2.INTER_LINEAR)
    # img_right = cv2.resize(img_right, (256, 256), interpolation=cv2.INTER_LINEAR)

    img_front = rearrange(img_front, 'h w c -> c h w') / 255.0  
    # img_left = rearrange(img_left, 'h w c -> c h w') / 255.0
    # img_right = rearrange(img_right, 'h w c -> c h w') / 255.0
    qpos = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)

    img_cnt += 1
    return dict(
        cam_high = img_front,
        # cam_left = img_left,
        # cam_right = img_right,
        qpos = qpos
    ), img_cnt


def load_hdf5(dataset_dir, dataset_name):
    # 加载hdf5数据集，重演数据
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs.get('sim', False)
        compressed = root.attrs.get('compress', False)
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        if 'effort' in root.keys():
            effort = root['/observations/effort'][()]
        else:
            effort = None
        action = root['/action'][()]
        base_action = root['/base_action'][()]
        master_action = root['/master_action'][()]
        
        action_mode_list = ["use_action", "use_master_action", "use_master_gripper_action"]
        action_mode = action_mode_list[2]
        
        if action_mode == "use_action":
            # 使用root['/action']
            pass
        elif action_mode == "use_master_action":
            # 使用root['/master_action']
            action = master_action
        elif action_mode == "use_master_gripper_action":
            # 使用root['/master_action']的第7维和第14维作为gripper动作，剩余维使用root['/action']
            action[..., 6] = master_action[..., 6]
            action[..., 13] = master_action[..., 13]

        # action[..., 6] += 0.002
        # action[..., 13] += 0.002
        
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        
        if compressed:
            compress_len = root['/compress_len'][()]

    if compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            # un-pad and uncompress
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list): # [:1000] to save memory
                image_len = int(compress_len[cam_id, frame_id])
                
                compressed_image = padded_compressed_image
                image = cv2.imdecode(compressed_image, 1)
                image_list.append(image)
            image_dict[cam_name] = image_list

    return qpos, qvel, effort, action, base_action, image_dict


def model_inference(args):
    """
    执行模型推理并通过 ROS 环境控制机器人执行动作。
    该函数的主要流程：
    1) 从给定的 checkpoint 文件加载训练好的策略（通过 hydra 实例化 LightningModule）。
    2) 将策略权重加载到模型，切换为 CUDA 并设置为评估模式。
    3) 初始化 ROS 操作器（RosOperator）和环境运行器（EnvRunner），并获取首帧图像以构造初始观测。
    4) 从指定的数据集路径加载预先记录的轨迹（qposs, qvels, efforts, actions, base_actions, image_dicts）。
    5) 在 torch.inference_mode() 下循环执行推理流程：按设定频率读取策略输出动作、遍历预录动作片段、将动作切分为左右臂（各 7 维）以及可选底盘速度，进行夹爪阈值裁剪，调用 env.step 执行动作，获取新观测并更新环境，直到达到最大发布步数或 ROS 关闭。
    注意事项与副作用：
    - 该函数通过 ROS 发布控制命令，会实际驱动硬件/仿真环境，请确保执行前环境安全。
    - 需要可用的 GPU（policy.cuda() 被调用），以及正确安装并配置的 ROS、PyTorch、Hydra 等依赖。
    - 若权重加载出现缺失或多余键（missing / unexpected），函数会打印信息并提前返回，不会进入推理循环。
    - 推理过程中会检查 rospy.is_shutdown() 用于安全退出。
    参数（通过 args 对象提供，必须包含以下属性）：
    - ckpt_dir (str): checkpoint 所在目录路径。
    - publish_rate (float/int): 推理循环的发布频率（Hz），用于 rospy.Rate。
    - max_publish_step (int): 最大发布步数，达到后退出循环。
    - use_robot_base (bool): 是否解析并发送底盘速度动作（action[14:16]）。
    - dataset_dir (str): 存放数据集的父目录路径。
    - task_name (str): 任务子目录名（用于拼接到 dataset_dir 下）。
    - episode_idx (int): 要加载的 episode 索引（用于组成 dataset_name = f'episode_{episode_idx}'）。
    - 其余：args 对象可包含其它自定义字段，函数会按需读取。
    外部依赖与期望函数/类：
    - torch, dill, hydra, pytorch_lightning.LightningModule
    - RosOperator, EnvRunner（必须实现 reset(), get_frame(), update_obs(), get_action(policy), step(...) 等方法）
    - get_model_input(frame, img_cnt)：将原始帧转换为模型输入观测并返回 (obs, img_cnt)
    - load_hdf5(path, dataset_name)：用于加载预录 trajectory，返回 (qposs, qvels, efforts, actions, base_actions, image_dicts)
    - rospy：用于时间控制与检测节点状态
    返回值：
    - None（函数通过副作用控制机器人并在自身内部终止）。
    异常处理：
    - 如果 checkpoint 权重与模型不兼容，会打印缺失/多余键并返回；其余运行时错误（如文件不存在、ROS 错误、网络/设备异常）将向上抛出，调用者应捕获处理。
    """
    
    # 1. 定义保存路径
    # save_dir  = Path('demo')
    # save_dir.mkdir(parents=True, exist_ok=True)
    # left_dir  = save_dir / 'left'
    # right_dir = save_dir / 'right'

    # left_dir.mkdir(parents=True, exist_ok=True)
    # right_dir.mkdir(parents=True, exist_ok=True)

    add_noise=False


    # 2 create env 创建env
    ros_operator = RosOperator(args)
    env = EnvRunner(ros_operator=ros_operator)
    env.reset()

    frame = env.get_frame()
    img_cnt = 0
    obs, img_cnt = get_model_input(frame, img_cnt)
    env.update_obs(obs)


    dataset_dir = args.dataset_dir
    episode_idx = args.episode_idx
    task_name   = args.task_name
    dataset_name = f'episode_{episode_idx}'
    qposs, qvels, efforts, actions, base_actions, image_dicts = load_hdf5(os.path.join(dataset_dir, task_name), dataset_name)

    actions = action_topp(actions, num=0)

    with torch.inference_mode():
        publish_step = 0
        rate = rospy.Rate(args.publish_rate)
        while True and not rospy.is_shutdown():
            publish_step = publish_step + 1
            step=0
            if publish_step > args.max_publish_step:
                break
            
            # now_actions = actions[max(0, (publish_step-1)*8-4):min(publish_step*8, len(actions)-1)]

            for action in actions:
                if rospy.is_shutdown():
                    break

                if step%100 == 0:
                    time.sleep(1)

                left_action = action[:7] 
                right_action = action[7:14]

                if add_noise:
                    noise = 0.005 * np.random.randn(*right_action.shape)
                    noise[-1] = 0
                    right_action = right_action + noise

                
                # left_action[-1] = left_action[-1] if left_action[-1] > 0.025 else 0
                # right_action[-1] = right_action[-1] if right_action[-1] > 0.025 else 0
                vel_action = None
                if args.use_robot_base:
                    vel_action = action[14:16]
                env.step(left_action, right_action, vel_action)

                frame = env.get_frame()
                obs, img_cnt = get_model_input(frame, img_cnt)
                env.update_obs(obs)

                rate.sleep()
                step+=1

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
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.',  default="/home/agilex/data_wx/playing_card_delivery/", required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',default=1, required=False)

    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='aloha_mobile_dummy', required=False)
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=250, required=False)

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

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    model_inference(args)


if __name__ == '__main__':
    main()