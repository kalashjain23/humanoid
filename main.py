import mujoco
import mujoco.viewer


def print_joint_and_actuators(model):
    for i in range(model.njnt):
        name = model.joint(i).name
        joint_type = model.joint(i).type # 0=free, 1=ball, 2=slide, 3=revolute
        joint_range = model.jnt_range[i]
        print(f"Joint {i}: {name}, type={joint_type}, range={joint_range}")
        
    for i in range(model.nu):
        print(f"Actuator {i}: {model.actuator(i).name}")
        
def bent_pose(data):
    data.qpos[18] = -0.5
    data.qpos[21] = 0.5
    data.qpos[22] = -0.3
    
    data.qpos[24] = -0.5
    data.qpos[27] = 0.5
    data.qpos[28] = -0.3
    
    return data

def main():
    model = mujoco.MjModel.from_xml_path("booster_t1/scene.xml")
    data = mujoco.MjData(model)
    # mujoco.viewer.launch(model, data)
    # print_joint_and_actuators()
        
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            bent_pose_data = bent_pose(data)
            
            mujoco.mj_step(model, bent_pose_data)
            viewer.sync()


if __name__ == "__main__":
    main()
