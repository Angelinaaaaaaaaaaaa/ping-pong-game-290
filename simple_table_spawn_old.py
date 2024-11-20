import mujoco_py
from mujoco_py import load_model_from_xml, MjSim, MjViewer, cymj
import numpy as np
import sys
# import mujoco
import mujoco
from mujoco import load
print(sys.executable)
print(sys.path)

# MuJoCo XML definition for the table tennis setup with bounce properties
xml = """
<mujoco model="table_tennis">
    <include file="iiwa14.xml"/>
    <compiler angle="radian" />
    <option timestep="0.001" gravity="0 0 -9.81" />
    <worldbody>
        <!-- Ground -->
        <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>

        <!-- Table -->
        <body name="table" pos="0 0 0.76">
            <!-- Table surface -->
            <geom name="table_top" type="box" size="1.37 0.7625 0.02" rgba="0 0 1 1" friction="0.2 0.2 0.1" solref="0.01 0.1" solimp="0.9 0.999 0.001" />
        </body>

        <body name="net" pos="0 0 0.78" euler="0 0 0"> <!-- Position and rotate net -->
                <!-- Net surface -->
                <geom name="net_geom" type="box" size="0.01 0.7625 0.15" rgba="1 1 1 1" 
                      friction="0 0 0" solref="0.00001 1" solimp="0.99 0.9999 0.00001" />
            </body>
            
        <!-- Ball -->
        <body name="ball" pos="0.5 0 2">
            <freejoint name="haha"/>
            <geom name="ball_geom" type="sphere" size="0.02" mass="0.0027" rgba="1 0.5 0 1" 
                  friction="0.001 0.001 0.001" solref="0.01 0.05" solimp="0.9 0.999 0.001" />
        </body>

        
        
    </worldbody>

    <actuator>
        <!-- You can add actuators here for paddles or other movements -->
    </actuator>
</mujoco>
"""

# Load the model and create the simulation
model = mujoco_py.load_model_from_path("mujoco_menagerie/kuka_iiwa_14/iiwa14.xml")
sim = MjSim(model)

# Create a viewer to visualize the simulation
viewer = MjViewer(sim)

# Initialize the velocity of the ball to simulate a bouncing drop
ball_id = sim.model.body_name2id("ball")
print(ball_id,sim.data)
sim.data.qvel[0] = -2  # Set downward velocity for bounce

print(sim.data)
# Simulate for a few seconds
for _ in range(10000):
    sim.step()
    viewer.render()
    # print(sim.data.get_joint_qpos("haha"))
    ball_pose = sim.data.get_joint_qpos("haha")
    if ball_pose[0] < -2:
        sim.data.qvel[0] = -2
        sim.data.qvel[1] = 0
        sim.data.qvel[2] = 0
        sim.data.qpos[0] = 0.5
        sim.data.qpos[1] = 0
        sim.data.qpos[2] = 2
        # sim.data.qvel[0] = 2
    # print("haha")
