import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from mujoco_env_only_kuka import KukaTennisEnv
from stable_baselines3 import PPO
import numpy as np
from scipy.spatial.transform import Rotation as R

class PoseSubscriberPublisher(Node):

    def __init__(self):
        super().__init__('pose_sub_pub')

        # Initialize subscriber to listen for PoseStamped messages
        self.subscription = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/bat_close/pose',
            self.pose_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        # # Initialize publisher to publish current_target_pose
        # self.pose_publisher = self.create_publisher(Float64MultiArray, '/current_target_pose', 10)
        
        # Initialize timer to publish at 100 Hz (1/100 seconds)
        self.timer = self.create_timer(0.03, self.publish_current_target_pose)

        # Initialize the target pose variable (x, y, z, qx, qy, qz, qw)
        self.current_target_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        self.env = KukaTennisEnv(proc_id=2)
        self.model = PPO.load("logs/best_model/best_model")
        self.obs, _ = self.env.reset()

    def Rot_x(self,x_deg) :
        x = np.radians(x_deg)
        return np.array([[1, 0, 0],
                        [0, np.cos(x), -np.sin(x)],
                        [0, np.sin(x), np.cos(x)]])

    def Rot_y(self,y_deg) :
        y = np.radians(y_deg)
        return np.array([[np.cos(y), 0, np.sin(y)],
                        [0, 1, 0],
                        [-np.sin(y), 0, np.cos(y)]])
    
    def Rot_z(self,z_deg) :
        z = np.radians(z_deg)
        return np.array([[np.cos(z), -np.sin(z), 0],
                        [np.sin(z), np.cos(z), 0],
                        [0, 0, 1]])

    def pose_callback(self, msg: PoseStamped):
        # Update the current_target_pose with the received position and quaternion
        position = msg.pose.position
        orientation = msg.pose.orientation
        curr_q = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
        r = R.from_quat(curr_q)
        rot1 = np.array([[-1,0,0],[0,0,1],[0,1,0]])
        rot2 = np.array([[0,0,1],[-1,0,0],[0,-1,0]])
        Rot_new = rot2@r.as_matrix()@rot1
        q = R.from_matrix(Rot_new).as_quat()
        self.current_target_pose = np.array([
            position.z, -position.x, -position.y, 
            q[0], q[1], q[2], q[3]
        ])
        self.env.set_target_pose(self.current_target_pose)
        self.get_logger().info(f'Updated target pose: {self.current_target_pose}')

    def publish_current_target_pose(self):
        action, _ = self.model.predict(self.obs, deterministic=True)
        # action = env.get_expert_cmd()
        self.obs, reward, done, _, info = self.env.step(action)
        # print(i, reward)
        self.env.render()
        # if done:
            # print("Reset requested")
            # self.obs, _ = self.env.reset()
        # if done:
        #     obs, _  = env.reset()


def main(args=None):
    rclpy.init(args=args)
    node = PoseSubscriberPublisher()
    rclpy.spin(node)
    node.env.close()
    node.destroy_node()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()




