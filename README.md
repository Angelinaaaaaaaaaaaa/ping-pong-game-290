# EE206 class project

Control Kuka iiwa R820 arm to play ping pong

<p float="left">
  <img src="iiwa_14.png" width="400">
</p>

## SSH into 337 cory lab machine

```bash 
ssh dvij@10.41.45.23
```

passwd: dvij123

## Installation on local machine

1. Clone the repo

    ```bash
   git clone https://github.com/dvij542/ee206-project
    ```

3. Install dependencies

    ```bash
   pip install -r requirements.txt
    ```

    OR 
    
    With conda:-
    
    ```bash
   conda env create --name ee206 --file=environment.yml
    ``` 

5. Install ROS Noetic for ubuntu 20.04/ compatible ROS1 version based on your system: [Link](https://wiki.ros.org/noetic/Installation/Ubuntu)

## Running

1. (optional) To run training on any environment, specify the environment in train.py and run:-

    ```bash
   python3 train.py
    ```
    
    NOTE: This is not required as the trained models for each environemnt is already present under logs/

3. To launch any environment and check it's performance, run:-

    ```bash
   python3 eval.py [--gantry] [--table] [--render] [--ik_rl]
    ```
  
    **a.** Enabling 'gantry' option will enable XY movement of the base. Note that we do not have a gantry for the robot yet (It has been ordered and may arrive in a month but should be fine without that)
    
    **b.** Enabling 'table' option will add the ping pong table to the environment with random generated ball throws towards the robot. It will also generate a target pose for the robot if able to be achieved in time, it will be able to return the ball to the center of the other side of the table. It woukd be in most cases
    
    **c.** Enabling 'render' option will render the environment
    
    **d.** Enabling 'ik_rl' option will use a more stable controller based on RL based IK solving to generate target robot pose. This is available for only without gantry for now
    
    To launch environment with gantry and table, run:-
    
    ```bash
   python3 eval.py --gantry --table --render
    ```
    
    To launch environment without gantry and with table, run:-
    
    ```bash
   python3 eval.py --table --render
    ```
    
    To launch environment without gantry and with table and with a more stable RL based IK solver, run:-
    
    ```bash
   python3 eval.py --table --ik_rl --render
    ```

5. To launch gazebo environment provided by Kuka iiwa and control the robot with our solvers, open 2 new terminals and follow these steps:-

    **a.** On terminal 1, source ROS1 noetic OR the distro installed on your system and then launch gazebo sim
    
    ```bash 
    source /opt/ros/noetic/setup.bash
    source lbr_fri_ros_ws/devel/setup.bash
    roslaunch lbr_moveit moveit_planning_execution.launch model:=iiwa14 sim:=true 
    ```
    
    **b.** On terminal 2, source ROS1 noetic OR the distro installed on your system and then run eval_ros.py
    
    ```bash 
    source /opt/ros/noetic/setup.bash
    source lbr_fri_ros_ws/devel/setup.bash
    python3 eval_ros.py [--ik_rl] [--ik] 
    ```

    Before running with ik option, you need to chamnge the namespace as follows:-

    ```
    export ROS_NAMESPACE=lbr
    ```
    
    For running with mocap enabled for tele-oping, add --mocap option. Also, need to run the following to talk to both the real robot and the mocap

    ```
    sudo ip addr add 172.31.1.148/16 dev enp3s0
    ```

    No options enabled will run the basic trained RL environment which may not run very well due to sim-to-real transfer. 'ik_rl' option will enable using RL based IK solver which will be more stable. 'ik' will enable traditional IK based solver. To read pose from mocap, run :-

    ```
    roslaunch vrpn_client_ros sample.launch
    ```

## TODO:-

1. (Setup ROS1 based twin simulator in mujoco)

    **a.** Subscribe to '\ball_pose' topic containing ball position and velocity. Write a separate node to predict the ball trajectory and target pose for the robot (Refer to moujoco_env_kuka_with_table.py.
    
    **b.** Control the robot through '/lbr/PositionJointInterface_trajectory_controller/command' topic. Write a separate node which subscribes to '/lbr/PositionJointInterface_trajectory_controller/command' topic and passes the commands to mujoco sim and it also publishes the joint angle and joint velocities as robot state to '/lbr/PositionJointInterface_trajectory_controller/state'. It should also publish to '/ball_pose' to get the ground truth ball pose. This piece will be replaced by the node that Emily and Max are working on. 
    
    **c.** Something like eval_ros.py can be used to communicate with (b) to control the robot through '/lbr/PositionJointInterface_trajectory_controller/command' topic
