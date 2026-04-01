import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import math
from gymnasium import spaces
# from gym.utils import seeding
import gymnasium as gym
import mujoco_viewer
from scipy.spatial.transform import Rotation as R

TABLE_SHIFT = 1.5
target_point = np.array([TABLE_SHIFT+1.37/2.,0.38,0.56])
PREPARATION_TIME = -1
DT = 0.01
# MuJoCo XML definition with Franka Panda robot and table tennis setup
xml = """
<mujoco model="table_tennis">
    <include file="iiwa14_gantry.xml"/>
    <compiler angle="radian" />
    <option timestep="0.01" gravity="0 0 -9.81" />
    <worldbody>
        <!-- Ground -->
        <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="vis" pos="0 0 1.26" quat="0 0.7068252 0 0.7073883">
            <geom name="cylinder" type="cylinder" pos="0.2 0 0" size="0.10 0.015" rgba="0 1 0 0.3" contype="0" conaffinity="0"/>
            <geom name="handle" type="cylinder" pos="0.05 0 0" size="0.02 0.05" quat="0 0.7068252 0 0.7073883" rgba="0 0 1 0.3" contype="0" conaffinity="0"/>
        </body>
        <!-- Table -->
        <body name="table" pos="1.5 0 0.56">
            <!-- Table surface -->
            <geom name="table_top" type="box" size="1.37 0.7625 0.02" rgba="0 0 1 1" friction="0.2 0.2 0.1" solref="0.04 0.1" solimp="0.9 0.999 0.001" />
        </body>

        <body name="net" pos="1.5 0 0.57" euler="0 0 0"> <!-- Position and rotate net -->
            <!-- Net surface -->
            <geom name="net_geom" type="box" size="0.01 0.6625 0.1" rgba="1 1 1 1" friction="0 0 0" solref="0.00001 1" solimp="0.99 0.9999 0.00001" />
        </body>
            
        <!-- Ball -->
        <body name="ball" pos="2 0 2">
            <freejoint name="haha"/>
            <geom name="ball_geom" type="sphere" size="0.02" mass="0.0027" rgba="1 0.5 0 1" 
                  friction="0.001 0.001 0.001" solref="0.04 0.05" solimp="0.9 0.999 0.001" />
        </body>
    </worldbody>
</mujoco>
"""

class KukaTennisEnv(gym.Env):
    def __init__(self,proc_id=0,history=4):
        super(KukaTennisEnv, self).__init__()
        self.history = history  
        # Load the MuJoCo model
        self.model = mj.MjModel.from_xml_string(xml)  # Use your actual MuJoCo XML path
        self.data = mj.MjData(self.model)

        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(9,), dtype=np.float32)  # Adjust based on your actuator count
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9 + 9 + 6 + 9*history + 1,), dtype=np.float32)

        # Simulation time step
        self.sim_dt = self.model.opt.timestep

        self.ep_no = 0
        self.viewer = None
        self.last_racket_pos = np.zeros(3)
        self.max_episode_steps = 200
        self.current_step = 0
        self.orientation_K = 10.0
        self.dist_k = 10.0
        self.prev_reward = 0.
        self.tolerance_range = [2.5,1.0]
        self.tolerance_exp = 12_000_000/256
        self.total_steps = 0
        self.proc_id = proc_id
        self.prev_actions = np.zeros((history,9))
        self.last_qvel = np.zeros(3)
        self.last_qpos = np.zeros(3)
        self.side_target = 1.
        self.curr_target = np.array([0.,0.,0.,0.,0.,0.,1.])
        
    def set_target_pose(self,pose):
        # print("Req:", pose)
        self.curr_target = pose
        self.update_vis_pose(self.curr_target)

    def update_vis_pose(self,pose):
        # Update the cylinder geom position
        target_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'vis')
        # self.data.geom_xpos[target_geom_id] = np.array(pose)
        self.model.body_pos[target_geom_id] = pose[:3]
        self.model.body_quat[target_geom_id] = pose[[6,3,4,5]]


    def reset_ball_throw(self):
        # print(self.data.body('ball').cvel)
        start_pos = np.array([1.8+TABLE_SHIFT+np.random.uniform(-0.2,0.2),np.random.uniform(-0.74,0.74),0.9])
        
        initial_velocity = np.array([-4.5+np.random.uniform(-0.5,0.5), 0., 2., 0., 0., 0.])
        v_contact = -np.sqrt(initial_velocity[2]*initial_velocity[2] + 2*9.81*0.34)
        t_contact = -(v_contact - initial_velocity[2])/9.81
        vy_min = (-0.65-start_pos[1])/t_contact
        vy_max = (0.65-start_pos[1])/t_contact
        initial_velocity[1] = np.random.uniform(vy_min,vy_max)
        t_ = 0.12
        
        self.data.qpos[-7:-4] = start_pos+ initial_velocity[:3]*t_ + 0.5*np.array([0.,0.,-9.81])*t_*t_
        self.start_pos = start_pos#+ initial_velocity[:3]*t_ + 0.5*np.array([0.,0.,-9.81])*t_*t_
        # print(start_pos,self.data.qpos[-7:-4])
        self.data.qvel[-6:] = initial_velocity - np.array([0.,0.,9.81,0.,0.,0.])*t_
        self.start_vel = initial_velocity #- np.array([0.,0.,9.81,0.,0.,0.])*t_
        # print(self.data.qvel[-6:],initial_velocity)
        self.side_target = np.random.choice([-1.,1.])
    
    def set_next_checkpoint(self,vel,position):
        self.next_checkpoint_vel = np.array(vel)
        self.next_checkpoint_pos = np.array(position)
        
    def step(self, action):
        self.prev_actions[:-1,:] = self.prev_actions[1:,:]
        self.prev_actions[-1,:] = action/10.
        self.current_step += 1
        self.total_steps += 1
        # Apply action to actuators
        # self.data.ctrl[:] = np.array(action) + np.array(self.data.qpos[:7])
        self.data.ctrl[:] = np.array(action[2:])/10. + np.array(self.data.qpos[2:9])
        # print("vel:",self.data.qvel)
        # print("pos:",self.data.qpos)
        self.data.qpos[0] += 15*action[0]/1000.
        self.data.qpos[1] += 15*action[1]/1000.
        self.data.qpos[0] = np.clip(self.data.qpos[0],-1.,-0.5)
        self.data.qpos[1] = np.clip(self.data.qpos[1],-1.,1.)
        # print(np.array(self.data.qvel[-6:-3]))

        # Check for collisions
        total_reward = 0.
        ncon = self.data.ncon  # Number of contacts
        done = False
        if ncon > 0:
            for i in range(ncon):
                contact = self.data.contact[i]
                geom1 = contact.geom1
                geom2 = contact.geom2
                # geom1 = self.model.geom_id2name(geom1)
                # geom2 = self.model.geom_id2name(geom2)
                # Retrieve the names of geom1 and geom2
                geom1 = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom1)
                geom2 = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom2)
                if geom2 is None or geom1 is None:
                    continue
                # print(f"Collision detected between geom {geom1} and geom {geom2}")

                if geom1 == "ball_geom" or geom2 == "ball_geom":
                    # print("Ball hit")
                    ball_vel = np.array(self.last_qvel)
                    # print(ball_vel)
                    ball_vel[2] = -ball_vel[2]
                    # ball_vel = np.array(self.next_checkpoint_vel)
                    self.data.qpos[-7:-4] = self.last_qpos + DT*ball_vel
                    self.data.qvel[-6:-3] = ball_vel
                    
                    # print(self.data.qvel[-6:])
                if (geom1 == "racket" and geom2=="ball_geom") or (geom2 == "racket" and geom1=="ball_geom"):
                    # print("Racket hit the ball")
                    # Reflect ball perpendicular to racket
                    ball_vel = np.array(self.last_qvel)
                    racket_rot = R.from_quat(self.data.body('tennis_racket').xquat[[1,2,3,0]]).as_matrix()
                    racket_pos = np.array(self.data.body('tennis_racket').xpos)
                    racket_vel = (racket_pos - self.last_racket_pos)/(DT)
                    self.last_racket_pos = racket_pos
                    # print(racket_rot)
                    racket_normal = racket_rot[:,2]
                    racket_speed_towards_normal = 0.#max(np.dot(racket_vel,racket_normal),0)/10.
                    # racket_vel = self.data.body('tennis_racket').xvel
                    ball_pos = self.data.body('ball').xpos
                    # print(ball_vel,racket_speed_towards_normal)
                    ball_vel = ball_vel - 2*np.dot(ball_vel,racket_normal)*racket_normal + 2*racket_speed_towards_normal*racket_normal
                    ball_vel_table = np.sqrt(max(ball_vel[2]**2 + 2*9.81*(ball_pos[2]-0.56),0.))
                    t_hit = (ball_vel_table + ball_vel[2])/9.81
                    x_table = ball_pos[0] + ball_vel[0]*t_hit
                    y_table = ball_pos[1] + ball_vel[1]*t_hit
                    x_diff = x_table - target_point[0]
                    y_diff = y_table - self.side_target*target_point[1] # Left side
                    dist_from_target = np.sqrt(x_diff**2 + y_diff**2)
                    # total_reward += 50*np.exp(-2*abs(ball_vel[0]-4.5))
                    t1 = (ball_pos[0]+(1.8-TABLE_SHIFT))/4.5
                    s = ball_pos[2]-ball_vel[2]*t1 - 0.5*9.81*t1*t1
                    total_reward += 50*np.exp(-2*abs(ball_vel[0]-4.5))
                    total_reward += 50*np.exp(-4*abs(s-0.9))
                    if self.proc_id==0 :
                        total_reward += 100*np.exp(-dist_from_target)*np.exp(-self.total_steps/self.tolerance_exp)
                    if x_table > TABLE_SHIFT and x_table < TABLE_SHIFT+1.37 and y_table > -0.75 and y_table < 0.75 :
                        print("Returned successfully",self.side_target)
                        total_reward += 100*np.exp(-2*dist_from_target)
                    # if self.proc_id == 1 :
                    #     print("Reward:",total_reward)
                    # if self.proc_id <= 1 :
                    #     done = True
                    self.data.qpos[-7:-4] = self.last_qpos + 2*DT*ball_vel
                    self.data.qvel[-6:-3] = ball_vel
                    
        self.last_qvel = np.array(self.data.qvel[-6:-3])
        self.last_qpos = np.array(self.data.qpos[-7:-4])
        mj.mj_step(self.model, self.data)
        # self.calc_target_racket_pose()
        if self.current_step < PREPARATION_TIME :
            obs = np.float32(np.concatenate([self.data.qpos[:9], self.data.qvel[:9],self.start_pos,self.start_vel[:3],self.prev_actions.flatten(),np.array([self.side_target])]))
        else :
            obs = np.float32(np.concatenate([self.data.qpos[:9], self.data.qvel[:9],self.data.body('ball').xpos,self.data.qvel[-6:-3],self.prev_actions.flatten(),np.array([self.side_target])]))
        # print(self.data.qpos)
        # Calculate reward and done
        # print(total_reward)
        self.update_target_racket_pose(y_target=self.side_target*0.38)
        reward, done_ = self._calculate_reward()
        # print(reward)
        if done_ :
            if self.proc_id <= 1 :
                done = True
            
            # if self.proc_id == 1 :
            #     print("Ended at ", self.current_step)
        else :
            total_reward += 30*(0.9-np.sum(np.abs(self.prev_actions[-1,:])))/(1800*0.1)
            if reward is not None :
                # print(self.prev_reward)
                if self.current_step > 3 and not done and abs(reward - self.prev_reward) < 3. :
                    total_reward += reward - self.prev_reward 
                self.prev_reward = reward
        
        # print(self.current_step)
        if self.current_step == PREPARATION_TIME :
            self.data.qpos[-7:-4] = self.start_pos + self.start_vel[:3]*0.12 + 0.5*np.array([0.,0.,-9.81])*0.12*0.12
            self.data.qvel[-6:] = self.start_vel - np.array([0.,0.,9.81,0.,0.,0.])*0.12

        end_effector_pos = self.data.body('tennis_racket').xpos
        end_effector_quat = self.data.body('tennis_racket').xquat[[1,2,3,0]]
        diff_pos = self.curr_target[:3] - end_effector_pos
        r_current = R.from_quat(end_effector_quat)
        r_target = R.from_quat(self.curr_target[3:7])
        diff_quat = r_target*r_current.inv()
        diff_quat = diff_quat.as_quat()
        
        if self.current_step >= self.max_episode_steps:
            self.current_step = 0
            done = True
            # if self.proc_id == 1 :
            #     print("Episode ended",self.proc_id)
                
        # print(total_reward)
        return obs, total_reward, done, False, {'diff_pos':diff_pos,'diff_quat':diff_quat,'target':self.curr_target}

    def reset_target(self):
        self.curr_target = np.array([0.,0.,0.,0.,0.,0.,0.])
        self.curr_target[0] = np.random.uniform(-0.2,0.2)
        self.curr_target[1] = np.random.uniform(-0.5,0.5)
        self.curr_target[2] = np.random.uniform(0.75,1.05)
        xr,yr,zr = np.random.uniform(-1,1,3)*0.5
        z_axis = np.array([1.,yr,zr])
        z_axis = z_axis/np.linalg.norm(z_axis)
        theta_z = np.arctan2(self.curr_target[2],self.curr_target[1])+xr*0.5
        x_axis = np.array([0.,np.cos(theta_z),np.sin(theta_z)])
        x_axis[0] = (-x_axis[1]*z_axis[1] - x_axis[2]*z_axis[2])/z_axis[0]
        x_axis = x_axis/np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis,x_axis)
        Rot = np.array([x_axis,y_axis,z_axis]).T
        r = R.from_matrix(Rot)
        q = r.as_quat()
        self.curr_target[3:7] = q
        self.update_vis_pose(self.curr_target)
    
    def calc_target_racket_pose_(self,x_racket=0.,bounce_factor=1,table_z=0.56,g = -9.81,x_target=TABLE_SHIFT+1.37/2.,y_target=0.):
        # Get ball position and velocity
        ball_pos = self.data.body('ball').xpos
        ball_vel = np.zeros((3,1))
        ball_vel[:,0] = self.data.qvel[-6:-3]
        if ball_pos[0] < x_racket-0.3 :
            return None, True
        if ball_vel[0,0] > 0 or ball_vel[2,0] > 8. or ball_pos[2] < 0.56:
            return None, False
        # print("vel:",ball_vel[2,0])
        # x = 0.
        T = (x_racket - ball_pos[0])/ball_vel[0,0]
        y = ball_pos[1] + ball_vel[1,0]*T
        
        z = ball_pos[2] + ball_vel[2,0]*T + 0.5*g*T*T - 0.1
        v_future = ball_vel + np.array([[0.],[0.],[g*T]])
        if z<table_z :
            vz_bounce = -np.sqrt(-2*g*max(ball_pos[2]-table_z,0)+ball_vel[2,0]*ball_vel[2,0])
            t_bounce = (vz_bounce - ball_vel[2,0])/g
            t_remaining = T - t_bounce
            v_future[2,0] = -vz_bounce + g*t_remaining
            z = table_z + 0.5*g*t_remaining*t_remaining - vz_bounce*t_remaining*bounce_factor -0.1
        pos = np.array([x_racket,y,z])
        # print("Target:",pos)
        # Calculate racket orientation
        # All possible z_axis are of form [1,yr,zr]. Generate them with np.meshgrid on yr,zr from range -1,1
        yzr = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,0.1,100))
        yzr = np.array(yzr).reshape(2,-1).T
        xyzr = np.concatenate([np.ones((yzr.shape[0],1)),yzr],axis=1)
        xyzr = xyzr/np.linalg.norm(xyzr,axis=1)[:,None]
        # ball_vel[2,0] += g*T
        ball_vels = np.tile(v_future.T,(100*100,1)).T
        ball_reflected_vels = ball_vels - 2*np.sum(ball_vels*xyzr.T,axis=0,keepdims=True)*xyzr.T
        vz_hits = -np.sqrt(-2*g*max(pos[2]-table_z,0)+ball_reflected_vels[2]*ball_reflected_vels[2])
        t_hits = (vz_hits - ball_reflected_vels[2])/g
        x_hits = pos[0] + ball_reflected_vels[0]*t_hits
        y_hits = pos[1] + ball_reflected_vels[1]*t_hits
        costs = (x_hits-x_target)**2 + (y_hits-y_target)**2
        idx = np.argmin(costs)
        z_axis = xyzr[idx]
        # print("x:",x_hits[idx],"y:",y_hits[idx],"z:",z_axis,"t:",t_hits[idx])
        theta_z = np.arctan2(z,x_racket)
        x_axis = np.array([0.,np.cos(theta_z),np.sin(theta_z)])
        x_axis[0] = (-x_axis[1]*z_axis[1] - x_axis[2]*z_axis[2])/z_axis[0]
        x_axis = x_axis/np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis,x_axis)
        Rot = np.array([x_axis,y_axis,z_axis]).T
        r = R.from_matrix(Rot)
        q = r.as_quat()
        return np.concatenate([pos,q]), False

    def update_target_racket_pose(self,bounce_factor=1,table_z=0.56,g = -9.81,x_target=TABLE_SHIFT+1.37/2.,y_target=0.38,x_dis=-1.8+TABLE_SHIFT,z_dis=0.9,vx_dis=4.5,vy_dis=2.):
        # Get ball position and velocity
        ball_pos = self.data.body('ball').xpos
        ball_vel = np.zeros((3,1))
        ball_vel[:,0] = self.data.qvel[-6:-3]
        if ball_vel[0,0] > 0 or ball_pos[0] < 0.3 or ball_vel[2,0] > 8. or ball_pos[2] < table_z+0.1:
            return
        # print("vel:",ball_vel)
        # x = 0.
        v_table = -np.sqrt(max(ball_vel[2,0]*ball_vel[2,0] - 2*g*(ball_pos[2]-table_z),0.))
        x_table = ball_vel[0,0]*(v_table - ball_vel[2,0])/g + ball_pos[0]
        if x_table > 0 : 
            x_racket = np.linspace(-0.7,min(0.4,x_table),100)
            T = (x_racket - ball_pos[0])/ball_vel[0,0]
            y = ball_pos[1] + ball_vel[1,0]*T
            z = ball_pos[2] + ball_vel[2,0]*T + 0.5*g*T*T
            v_future = np.zeros((3,100))
            v_future[0,:] = ball_vel[0,0]
            v_future[1,:] = ball_vel[1,0]
            v_future[2,:] = ball_vel[2,0] + g*T
            vz_bounce = -np.sqrt(-2*g*max(ball_pos[2]-table_z,0)+ball_vel[2,0]*ball_vel[2,0])
            t_bounce = (vz_bounce - ball_vel[2,0])/g
            t_remaining = T - t_bounce
            v_future[2,:] = -vz_bounce + g*t_remaining
            z = table_z + 0.5*g*t_remaining*t_remaining - vz_bounce*t_remaining*bounce_factor
        else :
            x_racket = np.linspace(-0.7,0.4,100)
            T = (x_racket - ball_pos[0])/ball_vel[0,0]
            y = ball_pos[1] + ball_vel[1,0]*T
            z = ball_pos[2] + ball_vel[2,0]*T + 0.5*g*T*T
            v_future = np.zeros((3,100))
            v_future[0,:] = ball_vel[0,0]
            v_future[1,:] = ball_vel[1,0]
            v_future[2,:] = ball_vel[2,0] + g*T
            
        t_dis = (x_racket-x_dis)/vx_dis
        z_dis = z_dis + vy_dis*t_dis + 0.5*g*t_dis*t_dis
        errs = (z-z_dis)**2
        idx = np.argmin(errs)
        x_racket = x_racket[idx]
        y = y[idx]
        z = z[idx]
        v_future = v_future[:,idx:idx+1]
        pos = np.array([x_racket,y,z])
        # print("Target:",pos)
        # Calculate racket orientation
        # All possible z_axis are of form [1,yr,zr]. Generate them with np.meshgrid on yr,zr from range -1,1
        yzr = np.meshgrid(np.linspace(-1,1,100),np.linspace(-0.01,0.01,100))
        yzr = np.array(yzr).reshape(2,-1).T
        xyzr = np.concatenate([np.ones((yzr.shape[0],1)),yzr],axis=1)
        xyzr = xyzr/np.linalg.norm(xyzr,axis=1)[:,None]
        ball_vels = np.tile(v_future.T,(100*100,1)).T
        ball_reflected_vels = ball_vels - 2*np.sum(ball_vels*xyzr.T,axis=0,keepdims=True)*xyzr.T
        vz_hits = -np.sqrt(-2*g*max(pos[2]-table_z,0)+ball_reflected_vels[2]*ball_reflected_vels[2])
        t_hits = (vz_hits - ball_reflected_vels[2])/g
        x_hits = pos[0] + ball_reflected_vels[0]*t_hits
        y_hits = pos[1] + ball_reflected_vels[1]*t_hits
        costs = (x_hits-x_target)**2 + (y_hits-y_target)**2
        idx = np.argmin(costs)
        z_axis = xyzr[idx]
        # print("x:",x_hits[idx],"y:",y_hits[idx],"z:",z_axis,"t:",t_hits[idx])
        theta_z = np.arctan2(z,y)
        x_axis = np.array([0.,np.cos(theta_z),np.sin(theta_z)])
        x_axis[0] = (-x_axis[1]*z_axis[1] - x_axis[2]*z_axis[2])/z_axis[0]
        x_axis = x_axis/np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis,x_axis)
        Rot = np.array([x_axis,y_axis,z_axis]).T
        r = R.from_matrix(Rot)
        q = r.as_quat()
        pos[2] -= 0.2*np.sin(theta_z)
        pos[1] -= 0.2*np.cos(theta_z)
        # print("Updated target: ",pos,z_axis,x_hits[idx],y_hits[idx])
        self.set_target_pose(np.concatenate([pos,q]))

    def reset(self,seed=None):
        self.current_step = 0
        self.prev_actions = np.zeros((self.history,9))
        prev_robot_pos = np.array(self.data.qpos[:9])
        mj.mj_resetData(self.model, self.data)
        # self.reset_target()
        self.reset_ball_throw()
        self.ep_no += 1
        if self.ep_no%20 == 0 :
            for i in range(7):
                self.data.qpos[i] = np.random.uniform(-1.,1.)
            self.data.qpos[7] = np.random.uniform(-1.,-0.5)
            self.data.qpos[8] = np.random.uniform(-1.,1.)
        else :
            self.data.qpos[:9] = prev_robot_pos
        # target_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, 'vis')
        # print(self.data.geom_xpos[target_geom_id])
        mj.mj_forward(self.model, self.data)
        # print(self.data.geom_xpos[target_geom_id])
        # self.prev_reward, _ = self._calculate_reward()
        # print(end_effector_pos)
        
        # Return initial observation
        obs = np.float32(np.concatenate([self.data.qpos[:9], self.data.qvel[:9],self.data.body('ball').xpos,self.data.qvel[-6:-3],self.prev_actions.flatten(),np.array([self.side_target])]))
        end_effector_pos = self.data.body('tennis_racket').xpos
        end_effector_quat = self.data.body('tennis_racket').xquat[[1,2,3,0]]
        diff_pos = self.curr_target[:3] - end_effector_pos
        r_current = R.from_quat(end_effector_quat)
        r_target = R.from_quat(self.curr_target[3:7])
        diff_quat = r_target*r_current.inv()
        diff_quat = diff_quat.as_quat()
        
        info = {'diff_pos':diff_pos,'diff_quat':diff_quat,'target':self.curr_target}
        return obs, info

    def render(self, mode="human"):
        # return
        if not hasattr(self, 'viewer') or self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _calculate_reward(self):
        racket_pos = self.data.body('tennis_racket').xpos
        # print(racket_pos)
        racket_orientation = self.data.body('tennis_racket').xquat[[1,2,3,0]]
        
        pose_target, done = self.calc_target_racket_pose_(x_racket=racket_pos[0])
        if done or pose_target is None:
            return None, done
        # print(racket_orientation)
        target_point_ = np.array(target_point).copy()
        target_point_[1] = self.side_target*target_point_[1]
        target_dir = target_point_ - pose_target[:3]
        target_dir = target_dir/np.linalg.norm(target_dir) 
        r_current = R.from_quat(racket_orientation)
        # r_target = R.from_quat(pose_target[3:7])
        R_mat = r_current.as_matrix()
        z_axis = R_mat[:,2]
        # diff_quat_rel = r_target*r_current.inv()
        # error = diff_quat_rel.magnitude()
        # print("angle error: ",error,2*np.arcsin(np.linalg.norm(diff_quat_rel.as_quat()[:3])))
        # Implement your reward calculation
        reward = - self.dist_k*np.linalg.norm(racket_pos - pose_target[:3]) #- (self.orientation_K*error)
        reward += 10.*min(np.sum(target_dir*z_axis),0.5)
        return reward, done


    def _is_done(self):
        # Implement termination condition
        return False

# Initialize the GLFW window for rendering
def init_glfw():
    if not glfw.init():
        raise Exception("Unable to initialize GLFW")
    window = glfw.create_window(1280, 720, "MuJoCo Simulation", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Unable to create GLFW window")
    glfw.make_context_current(window)
    return window


if __name__ == "__main__":
    env = KukaTennisEnv()
    obs = env.reset()
    for i in range(1000):
        action = env.action_space.sample()*0  # Random action
        obs, reward, done, _, _ = env.step(action)
        # print(i,reward)
        env.render()
    env.close()
    