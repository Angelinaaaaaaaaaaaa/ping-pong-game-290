import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import math
import mujoco_viewer
TABLE_SHIFT = 1.5
# MuJoCo XML definition with Franka Panda robot and table tennis setup
xml = """
<mujoco model="table_tennis">
    <include file="iiwa14.xml"/>
    <compiler angle="radian" />
    <option timestep="0.004" gravity="0 0 -9.81" />
    <worldbody>
        <!-- Ground -->
        <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>
        
        <!-- Table -->
        <body name="table" pos="1.5 0 0.76">
            <!-- Table surface -->
            <geom name="table_top" type="box" size="1.37 0.7625 0.02" rgba="0 0 1 1" friction="0.2 0.2 0.1" solref="0.04 0.1" solimp="0.9 0.999 0.001" />
        </body>

        <body name="net" pos="1.5 0 0.78" euler="0 0 0"> <!-- Position and rotate net -->
                <!-- Net surface -->
                <geom name="net_geom" type="box" size="0.01 0.7625 0.15" rgba="1 1 1 1" 
                      friction="0 0 0" solref="0.00001 1" solimp="0.99 0.9999 0.00001" />
            </body>
            
        <!-- Ball -->
        <body name="ball" pos="2 0 2">
            <freejoint name="haha"/>
            <geom name="ball_geom" type="sphere" size="0.02" mass="0.0027" rgba="1 0.5 0 1" 
                  friction="0.001 0.001 0.001" solref="0.04 0.05" solimp="0.9 0.999 0.001" />
        </body>

        
        
    </worldbody>

    <actuator>
        <!-- You can add actuators here for paddles or other movements -->
    </actuator>
</mujoco>
"""


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

# Create the GLFW window
window = init_glfw()

# Create the GLFW window
# window = init_glfw()

def get_target_pos_at(x,curr_ball_pose,curr_ball_v,table_z,bounce_factor=1.0) :
    T = (x - curr_ball_pose[0])/curr_ball_v[0]
    y = curr_ball_pose[1] + curr_ball_v[1]*T
    g = -9.81
    z = curr_ball_pose[2] + curr_ball_v[2]*T + 0.5*g*T*T
    if z>table_z :
        return [x,y,z]
    else :
        # Ball bounces at z = table_z
        vz_bounce = -np.sqrt(-2*g*(curr_ball_pose[2]-table_z)+curr_ball_v[2]*curr_ball_v[2])
        t_bounce = (vz_bounce - curr_ball_v[2])/g
        t_remaining = T - t_bounce
        z = table_z + 0.5*g*t_remaining*t_remaining - vz_bounce*t_remaining*bounce_factor
        return [x,y,z]
  
# Load the model and data
model = mj.MjModel.from_xml_string(xml)
data = mj.MjData(model)

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

# Set initial velocity of the ball (forward in the X direction)
initial_velocity = np.array([-2.0, 0.0, 0.0])  # X-axis velocity of 2.0 m/s
ball_joint_index = -1
# print(jacp,jacr)
# print(np.linalg.pinv(jacp))
# exit(0)
# Set linear velocity (first 3 components are linear velocities for a free joint)
data.qvel[ball_joint_index * 6 : ball_joint_index * 6 + 3] = initial_velocity
# for i in range(100):
#     data.ctrl = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     mj.mj_step(model, data)
#     # print(data.body('tennis_racket').xquat)
#     # Convert quaternion to euler angles
#     q = data.body('tennis_racket').xquat
#     euler = euler_from_quaternion(q[0], q[1], q[2], q[3])
#     # print(data.qpos[:7])
# exit(0)
# Create a camera, options, and context for rendering
camera = mj.MjvCamera()
opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=1000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)



# Variables for interactive camera movement
last_x, last_y = 0, 0
button_left, button_right, button_middle = False, False, False

# Mouse button callback
def mouse_button_callback(window, button, action, mods):
    global button_left, button_right, button_middle
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)

# Mouse motion callback
def mouse_motion_callback(window, xpos, ypos):
    global last_x, last_y, button_left, button_right, button_middle

    dx = xpos - last_x
    dy = ypos - last_y

    dx*=0.5
    dy*=0.5

    last_x = xpos
    last_y = ypos

    if not button_left and not button_right and not button_middle:
        return

    # Rotate view if left mouse button is held
    if button_left:
        mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ROTATE_V, dx / 100, dy / 100, scene, camera)
    
    # Zoom if right mouse button is held
    if button_right:
        mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0, dy / 100, scene, camera)

    # Translate if middle mouse button is held
    if button_middle:
        mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_MOVE_V, dx / 300, dy / 300, scene, camera)

# Scroll callback for zoom
def scroll_callback(window, xoffset, yoffset):
    mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0, -0.02 * yoffset, scene, camera)

# Set callbacks for the window
glfw.set_mouse_button_callback(window, mouse_button_callback)
glfw.set_cursor_pos_callback(window, mouse_motion_callback)
glfw.set_scroll_callback(window, scroll_callback)
ball_joint_index = -1
kp = 0.5
# Simulation step and rendering function
def simulate_and_render(window, model, data):
    global last_x, last_y
    curr_itr = 0
    while not glfw.window_should_close(window):
        # Advance the simulation by one step
        curr_itr += 1
        mj.mj_step(model, data)
        # print(data.qpos)
        # print("haha", len(data.qpos))
        ball_pose = data.qpos[ball_joint_index*7:ball_joint_index*7+3]
        ball_v = data.qvel[ball_joint_index*6:ball_joint_index*6+3]
        target_pos = get_target_pos_at(0.,ball_pose,ball_v,0.76)
        # target_pos = np.array([-0.03,0.5,0.8])
        jacp = np.zeros((3, model.nv))  # Jacobian for translational velocity (3D vector)
        jacr = np.zeros((3, model.nv))  # Jacobian for rotational velocity (3D vector)
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'tennis_racket')
        curr_pos = data.body('tennis_racket').xpos
        d_pos = np.array(target_pos) - curr_pos
        # print(data.body('tennis_racket'),body_id)
        mj.mj_jac(model, data, jacp, jacr, data.body('tennis_racket').xpos, body_id)
        target_joint_vel = kp*np.dot(np.linalg.pinv(jacp),d_pos)
        target_joint_pos = np.array(data.qpos[:7]) + target_joint_vel[:7]
        data.ctrl = np.array(target_joint_pos)
        # print(curr_pos,target_pos)
        print(curr_itr)
        if data.qvel[ball_joint_index*6] > 0 or curr_itr>300:
        # if curr_itr%300 == 0:
            curr_itr = 0
            data.qvel[ball_joint_index*6] = -4
            data.qvel[ball_joint_index*6+1] = 0
            data.qvel[ball_joint_index*6+2] = 0
            data.qpos[ball_joint_index*7+0] = 0.8+TABLE_SHIFT
            data.qpos[ball_joint_index*7+1] = np.random.uniform(-0.5,0.5)
            data.qpos[ball_joint_index*7+2] = 1.3
        
        # Get the framebuffer size for rendering
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        # Update the scene
        mj.mjv_updateScene(model, data, opt, None, camera, mj.mjtCatBit.mjCAT_ALL, scene)

        # Render the scene
        mj.mjr_render(viewport, scene, context)

        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

# Run the simulation and rendering
simulate_and_render(window, model, data)

# Clean up and close the window
glfw.terminate()
