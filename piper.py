import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
import time
try:
    import termios
    import tty
except ImportError:
    import msvcrt
import rospy
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import os
import sys
import threading
from piper_control import PIPER
from piper_msgs.msg import PosCmd

piper_control = PIPER()

exit_flag = False

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class Arm_IK:
    """
    Arm_IK 类 — 基于 Pinocchio 与 CasADi 的机械臂逆向运动学求解器（含碰撞检测与可视化）
    简介
    ----
    Arm_IK 提供了一个用于 URDF 描述机械臂的逆向运动学（IK）求解器，集成了：
    - Pinocchio：正向/逆向运动学、动力学、几何碰撞检测
    - CasADi：符号化误差表达与数值优化（使用 IPOPT）
    - Meshcat：可视化（显示机器人、目标位姿、参考坐标系）
    该类对末端执行器目标位姿进行优化求解，同时支持自碰撞检测与简单的正则化项。
    主要功能
    ----
    - 构建并简化（锁定部分关节）机器人模型（基于 URDF）
    - 创建末端“ee”帧并在 Meshcat 中显示目标坐标系
    - 使用 CasADi 构造误差函数（基于 6D 对数映射）并形成约束优化问题
    - 求解 IK（ik_fun），并在求解后进行自碰撞检测
    - 提供便捷接口 get_ik_solution 通过位置与欧拉角直接求解并发送控制命令（与 piper_control 集成）
    初始化参数与依赖（在 __init__ 中隐含）
    ----
    - 需要有效的 URDF 路径以创建 pin.RobotWrapper
    - 依赖模块：pinocchio（pin）、casadi、casadi-pinocchio（cpin）、MeshcatVisualizer、mg（meshcat.geometry）等
    - 会构建：self.robot, self.reduced_robot, self.geom_model, self.geometry_data, self.vis（Meshcat 可视化）
    - 构建 CasADi 模型与变量：self.cmodel, self.cdata, self.cq, self.cTf, self.error
    - 构建并配置 CasADi Opti：self.opti、self.var_q、self.param_tf、目标函数与约束（关节上下限）
    关键方法
    ----
    ik_fun(target_pose, gripper=0, motorstate=None, motorV=None)
        根据目标齐次变换矩阵（4x4）求解逆向运动学。
        参数：
        - target_pose: 4x4 齐次变换矩阵（numpy array 或能被 Meshcat set_transform 接受的格式）
        - gripper: 标量或用于生成并联夹持器两个电机的简化数组（内部转为长度为2的数组）
        - motorstate: 初始关节角（用于初始化优化起点，可为空）
        - motorV: 关节速度（用于计算惯性/力相关项的占位，当前实现多为 0）
        返回：
        - sol_q: 求解得到的关节角数组（reduced_robot.model.nq）
        - tau_ff: 以 sol_q 为输入计算的前馈关节力矩（RNEA）
        - success_flag: 布尔值，True 表示无自碰撞且求解成功，否则为 False
        行为与异常：
        - 若 IPOPT 未收敛或异常抛出，方法捕获异常并返回 (None, '', False)
        - 内部会在 Meshcat 中显示目标帧与求解结果（若可视化已初始化）
    check_self_collision(q, gripper=np.array([0, 0]))
        使用 Pinocchio 的几何模块检测给定关节配置下的自碰撞情况。
        参数：
        - q: reduced_robot 的关节角（长度为 reduced_robot.model.nq）
        - gripper: 与机器人完整模型对应的夹爪关节值（长度与 robot 模型期望一致）
        返回：
        - collision: 布尔或碰撞信息（True 表示存在碰撞）
        说明：
        - 内部执行 forwardKinematics、updateGeometryPlacements、computeCollisions
    get_ik_solution(x, y, z, roll, pitch, yaw)
        便捷接口：根据位移 (x,y,z) 和欧拉角（roll,pitch,yaw）构造目标位姿并调用 ik_fun 求解。
        - 将欧拉角转换为四元数，构造 pin.SE3 目标并传递给 ik_fun
        - 若求解成功并通过碰撞检测，会调用外部 piper_control.joint_control_piper 发送关节控制命令
        - 若发生碰撞或求解失败，打印提示信息
    优化设计细节
    ----
    - 目标误差函数由位姿差（6D，即位置3 + 旋转3）给出，分配了位置与姿态权重（示例：位置权重 1.0、姿态权重 0.1）
    - 总代价为带权的位姿误差平方和 + 小的关节正则化项
    - 关节角以机器人 model 提供的 lower/upperPositionLimit 作为约束
    - 使用 IPOPT 求解器，内部默认最大迭代次数与容差可在 opts 中调整
    注意事项与使用建议
    ----
    - 需确保 URDF 路径和帧名（例如 "ee"）与实际一致，否则会抛出查找错误
    - Meshcat 可视化为可选，但本实现中会尝试初始化并加载模型；在无可视化环境可调整或禁用
    - gripper 在 reduced 与 full robot model 中映射需对应，check_self_collision 将把 reduced q 与 gripper 合并以用于完整模型的碰撞检测
    - 优化收敛依赖初始值 motorstate，建议对连续求解使用上一次解作初值以提高稳定性
    - 若需要平滑轨迹，可在目标函数中加入平滑项（示例在代码中以注释形式给出）
    - 返回的 tau_ff 为基于当前关节角和零加速度/外力计算的逆动力学前馈力矩，仅供参考
    示例（伪代码）
    ----
    # 构造
    ik = Arm_IK()
    # 求解单个目标位姿
    sol_q, tau, ok = ik.ik_fun(target_pose_homogeneous)
    # 或使用位置+欧拉角接口
    ik.get_ik_solution(0.5, 0.0, 0.2, 0.0, 0.0, 0.0)
    版权与依赖
    ----
    - 该类依赖 pinocchio、casadi 及 meshcat 等第三方库，使用前请确保已正确安装并能在当前环境中导入
    - 本 docstring 为对代码行为的说明，具体实现细节以源码为准
    """
    
    def __init__(self):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        
        # 载入 urdf 文件，创建机器人模型
        cur_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        last_path = os.path.dirname(current_dir)
        last_path = os.path.dirname(last_path)
        last_path = os.path.dirname(last_path)
        urdf_dir = os.path.join(last_path, 'piper_description/urdf/piper_description.urdf')
        # urdf_path = '/home/agilex/piper_ws/src/piper_description/urdf/piper_description.urdf'
        urdf_path = urdf_dir
    
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)

        # 锁定夹抓关节，创建简化机器人模型
        self.mixed_jointsToLockIDs = ["joint7",
                                      "joint8"
                                      ]

        # 创建简化机器人模型，锁定指定关节，参考配置设为全0
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        # 添加末端执行器“ee”帧，位于 joint6 处
        # q = quaternion_from_euler(0, -1.57, -1.57)
        q = quaternion_from_euler(0, 0, 0)
        self.reduced_robot.model.addFrame(
            pin.Frame('ee',
                      self.reduced_robot.model.getJointId('joint6'),
                      pin.SE3(
                          # pin.Quaternion(1, 0, 0, 0),
                          pin.Quaternion(q[3], q[0], q[1], q[2]),
                          np.array([0.0, 0.0, 0.0]),
                      ),
                      pin.FrameType.OP_FRAME)
        )

        self.geom_model = pin.buildGeomFromUrdf(self.robot.model, urdf_path, pin.GeometryType.COLLISION) # 创建碰撞几何模型
        n_geom = len(self.geom_model.geometryObjects)
        print("geom num:", n_geom) # 9
        
        # for i in range(4, 10): i和j的范围来源不明
        for i in range(4, 9):
            for j in range(0, 3):
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j)) # 添加碰撞对，避免检测某些部件间的碰撞
        self.geometry_data = pin.GeometryData(self.geom_model)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)

        # # Initialize the Meshcat visualizer  for visualization
        # 初始化 Meshcat 可视化器，用于可视化
        self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
        self.vis.initViewer(open=True)
        self.vis.loadViewerModel("pinocchio")
        self.vis.displayFrames(True, frame_ids=[113, 114], axis_length=0.15, axis_width=5)
        self.vis.display(pin.neutral(self.reduced_robot.model))

        # Enable the display of end effector target frames with short axis lengths and greater width.
        # 显示末端执行器目标帧，轴长度较短，宽度较大
        frame_viz_names = ['ee_target']
        FRAME_AXIS_POSITIONS = (
            np.array([[0, 0, 0], [1, 0, 0],
                      [0, 0, 0], [0, 1, 0],
                      [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
        )
        FRAME_AXIS_COLORS = (
            np.array([[1, 0, 0], [1, 0.6, 0],
                      [0, 1, 0], [0.6, 1, 0],
                      [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
        )
        axis_length = 0.1
        axis_width = 10
        for frame_viz_name in frame_viz_names:
            self.vis.viewer[frame_viz_name].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(
                        position=axis_length * FRAME_AXIS_POSITIONS,
                        color=FRAME_AXIS_COLORS,
                    ),
                    mg.LineBasicMaterial(
                        linewidth=axis_width,
                        vertexColors=True,
                    ),
                )
            )

        # Creating Casadi models and data for symbolic computing
        # 创建 Casadi 模型和数据，用于符号计算
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        # 创建符号变量
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # # Get the hand joint ID and define the error function
        # 获取手部关节 ID 并定义误差函数
        self.gripper_id = self.reduced_robot.model.getFrameId("ee")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )

        # Defining the optimization problem
        # 定义优化问题
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        # self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)
        # self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf))
        error_vec = self.error(self.var_q, self.param_tf)
        pos_error = error_vec[:3]  # 取前3个值为位置误差
        ori_error = error_vec[3:]  # 取后3个值为姿态误差
        # 设置位置和姿态的权重
        weight_position = 1.0  # 位置权重
        weight_orientation = 0.1  # 姿态权重
        # 总成本函数
        self.totalcost = casadi.sumsqr(weight_position * pos_error) + casadi.sumsqr(weight_orientation * ori_error)
        # 正则化项
        self.regularization = casadi.sumsqr(self.var_q)
        # self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last) # for smooth

        # Setting optimization constraints and goals
        # 设置优化约束和目标
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        # print("self.reduced_robot.model.lowerPositionLimit:", self.reduced_robot.model.lowerPositionLimit)
        # print("self.reduced_robot.model.upperPositionLimit:", self.reduced_robot.model.upperPositionLimit)
        self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization + 0.1 * self.smooth_cost) # for smooth

        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4
            },
            'print_time': False
        }
        self.opti.solver("ipopt", opts)

    def ik_fun(self, target_pose, gripper=0, motorstate=None, motorV=None):
        # IK 求解函数
        
        gripper = np.array([gripper/2.0, -gripper/2.0])
        if motorstate is not None:
            self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data)

        self.vis.viewer['ee_target'].set_transform(target_pose)     # for visualization

        self.opti.set_value(self.param_tf, target_pose)
        # self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

            if self.init_data is not None:
                max_diff = max(abs(self.history_data - sol_q))
                # print("max_diff:", max_diff)
                self.init_data = sol_q
                if max_diff > 30.0/180.0*3.1415:
                    # print("Excessive changes in joint angle:", max_diff)
                    self.init_data = np.zeros(self.reduced_robot.model.nq)
            else:
                self.init_data = sol_q
            self.history_data = sol_q

            self.vis.display(sol_q)  # for visualization

            if motorV is not None:
                v = motorV * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v,
                              np.zeros(self.reduced_robot.model.nv))

            is_collision = self.check_self_collision(sol_q, gripper)

            return sol_q, tau_ff, not is_collision

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            # sol_q = self.opti.debug.value(self.var_q)   # return original value
            return None, '', False

    def check_self_collision(self, q, gripper=np.array([0, 0])):
        # 碰撞检测函数
        pin.forwardKinematics(self.robot.model, self.robot.data, np.concatenate([q, gripper], axis=0))
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
        # print("collision:", collision)
        return collision

    def get_ik_solution(self, x,y,z,roll,pitch,yaw):
        # 通过位置和欧拉角获取逆运动学解
        
        q = quaternion_from_euler(roll, pitch, yaw)
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z]),
        )
        print(target)
        
        # 调用 ik_fun 求解逆运动学
        sol_q, tau_ff, get_result = self.ik_fun(target.homogeneous,0)
        # print("result:", sol_q)
        
        if get_result :
            # 无碰撞，发送控制命令
            piper_control.joint_control_piper(sol_q[0],sol_q[1],sol_q[2],sol_q[3],sol_q[4],sol_q[5],0)
        else :
            print("collision!!!")
    
class C_PiperIK():
    """
    C_PiperIK 类（逆运动学 ROS 接口）
    
    作用：监听message，输入消息后传给 Arm_IK 类进行逆运动学求解。
    
    概述
    ----
    C_PiperIK 封装了一个用于接收位置命令并触发机械臂逆运动学求解的简单 ROS 节点逻辑。
    该类负责：
    - 初始化 ROS 节点（节点名为 'inverse_solution_node'）
    - 创建 Arm_IK 实例（负责具体的逆解计算）
    - 启动一个后台订阅线程，订阅 'pin_pos_cmd' 话题，接收 PosCmd 消息并将数据传递给 Arm_IK.get_ik_solution
    
    属性
    ----
    arm_ik : Arm_IK
        用于计算逆运动学的实例。假定该类提供方法 get_ik_solution(x, y, z, roll, pitch, yaw)。
    _sub_thread : threading.Thread
        用于订阅 ROS 话题的后台守护线程（内部创建并启动）。
    
    方法
    ----
    __init__(self)
        初始化 ROS 节点并创建 Arm_IK 实例，启动订阅线程。
        注意：rospy.init_node 在整个进程中只应调用一次；如果在其他地方已初始化，重复调用可能产生警告或异常。
    SubPosThread(self)
        在独立线程中创建 rospy.Subscriber，订阅话题 'pin_pos_cmd'，消息类型为 PosCmd，并将回调函数设为 pos_cmd_callback。
        使用 rospy.spin() 进入 ROS 回调循环，直到节点关闭。
    pos_cmd_callback(self, msg)
        PosCmd 消息的回调处理函数。期望 msg 含有以下字段：
            - x, y, z : 位置分量（通常以米为单位）
            - roll, pitch, yaw : 姿态分量（通常以弧度或度为单位，具体以系统约定为准）
        回调功能：
            1. 从消息中提取 x, y, z, roll, pitch, yaw
            2. 调用 self.arm_ik.get_ik_solution(x, y, z, roll, pitch, yaw) 触发逆运动学求解
        返回：None
    
    行为与注意事项
    ----
    - 订阅线程设置为守护线程（daemon=True），当主线程退出时该线程随进程结束。
    - 假定 Arm_IK.get_ik_solution 是线程安全的；若非线程安全，应在外层加锁以防并发访问冲突。
    - 若 PosCmd 消息缺少字段或字段类型不符合预期，应在回调中加入必要的异常处理以避免线程崩溃。
    - rospy.spin() 会阻塞调用线程以处理 ROS 回调；因此将其放在独立线程中以避免阻塞主程序。
    - 单元测试或在非 ROS 环境中使用时，可通过注入模拟的 Arm_IK 或模拟消息来隔离依赖。
    示例（概念性说明）
    ----
    # 在 ROS 节点主程序中直接创建：
    piper = C_PiperIK()
    # 随后由 ROS 话题驱动 pos_cmd_callback 的执行
    """
    
    def __init__(self):
        # 创建 ROS 节点
        rospy.init_node('inverse_solution_node', anonymous=True)
        # 创建Arm_IK实例
        self.arm_ik = Arm_IK()
        
        # 启动订阅线程
        sub_pos_th = threading.Thread(target=self.SubPosThread, daemon=True)
        sub_pos_th.daemon = True # 设置为守护线程,如果主线程退出,该线程会自动退出
        sub_pos_th.start()
    
    def SubPosThread(self):
        # 创建订阅者，监听PosCmd类型的消息
        rospy.Subscriber('pin_pos_cmd', PosCmd, self.pos_cmd_callback)
        rospy.spin() # 阻塞当前线程，直到节点关闭

    def pos_cmd_callback(self, msg):
        # 获取PosCmd类型消息中的数据
        x = msg.x
        y = msg.y
        z = msg.z
        roll = msg.roll
        pitch = msg.pitch
        yaw = msg.yaw

        # 调用Arm_IK类的逆解函数
        self.arm_ik.get_ik_solution(x, y, z, roll, pitch, yaw)

def key_listener():
    """
    等待用户按下 'q' 键，并触发程序退出信号。
    此函数会监控键盘输入，当用户输入字母
    'q'（不区分大小写）时，将模块级别的全局变量 `exit_flag`
    设置为 True，并打印 "exit..."。该函数设计为在后台运行（例如，在一个专用线程中），因为它在等待输入时会阻塞。
    
    Behavior：
    - 在 Windows 上（os.name == 'nt'）：使用 msvcrt.kbhit() 和 msvcrt.getch()
        来轮询并读取单个按键，无需按 Enter 键。
    - 在 POSIX 系统上：通过 tty.setcbreak() 将终端切换到 cbreak 模式，
        并使用 sys.stdin.read(1) 读取单个字符；终端设置会在 finally 块中恢复，
        以避免使终端处于非规范状态。
        
    Side effects：
    - 当检测到 'q' 或 'Q' 时，将全局变量 `exit_flag` 设置为 True。
    - 退出时向标准输出打印 "exit..."。
    - 在 POSIX 系统上临时修改终端状态（退出时恢复）。
    
    Return：
    - 无
    
    Usage notes and caveats：
    - 该函数会一直阻塞，直到按下 'q' 键。如果主程序需要同时继续运行，
        请将其放在单独的线程中执行。
    - 对全局变量 `exit_flag` 的修改未进行同步。如果其他线程同时读取或写入此标志，
        考虑使用 threading.Event 或其他同步原语以确保线程安全。
    - 在 POSIX 系统上，从 sys.stdin.read(1) 读取数据会一直阻塞，直到有输入可用；
        这可能会与重定向或非交互式的标准输入产生冲突。
    - 特定于系统的模块（msvcrt、termios、tty）如果不可用或终端操作失败，
        可能会引发异常；调用方可能需要根据情况适当处理这些异常。
    """

    global exit_flag
    if os.name == 'nt':
        while True:
            if msvcrt.kbhit():
                if msvcrt.getch().lower() == b'q':
                    exit_flag = True
                    print("exit...")
                    break
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                if sys.stdin.read(1).lower() == 'q':
                    exit_flag = True
                    print("exit...")
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

if __name__ == "__main__":
    piper_ik = C_PiperIK()
    print("Press 'q' to quit")
    listener_thread = threading.Thread(target=key_listener, daemon=True)
    listener_thread.start()
    while not exit_flag:
        time.sleep(0.1)