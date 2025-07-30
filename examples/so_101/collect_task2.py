from copy import copy

import numpy as np
from tqdm import trange
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.factory import make_policy
from lerobot.policies.factory import make_policy_config
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
import torch
import imageio
import genesis as gs
import gym_genesis
import gymnasium as gym
from contextlib import nullcontext

env = gym.make(
    "gym_genesis/CubeStack-v0",
    enable_pixels=True,
    camera_capture_mode="global",
    strip_environment_state=False,
    num_envs=3 # this will be ignore, nothing is batched now
)
env = env.unwrapped

#     return path
def expert_policy_v22(robot, obs, stage):
    """
    Expert policy with extra XY correction before release and retreat.
    """
    eef = robot.get_link("gripper")
    quat = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device)
    
    
    cube1_pos = obs["environment_state"][:3]
    cube2_pos = obs["environment_state"][11:14]

    grip_open = 0.5
    grip_closed = 0.1
    z_offset = 0.18
    correction_xy = torch.tensor([-0.005, 0.02], device=gs.device)
    # Compensate for gripper's site offset in Z
    gripper_offset_z = -0.0981  # from <site name="gripper" ... pos="..." />

    # Correction to align center of gripper with cube2
    gripper_xy_correction = torch.tensor([-0.005, 0.02], device=gs.device)


    if stage == "hover":
        target_pos = cube1_pos + torch.tensor([-0.01, 0.0, 0.25], device=gs.device)
        grip_val = grip_open
    elif stage == "wake":
        current_pos = eef.get_pos()
        target_pos = current_pos + torch.tensor([0.0, 0.0, 0.25], device=gs.device)
        grip_val = grip_open
    elif stage == "grasp":
        target_pos = cube1_pos + torch.tensor([-0.01, 0.0, 0.045], device=gs.device)
        grip_val = grip_closed

    elif stage == "lift":
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.28], device=gs.device)
        grip_val = grip_closed

    elif stage == "place":
        target_pos = cube2_pos + torch.tensor([0.0, 0.0, z_offset], device=gs.device)
        grip_val = grip_closed

    elif stage == "position_align":
        target_xy = cube2_pos[:2] + gripper_xy_correction
        target_z = cube2_pos[2] + z_offset - gripper_offset_z
        target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_closed


    elif stage == "release":
        target_xy = cube2_pos[:2]  # stay centered
        target_z = cube2_pos[2] + z_offset - gripper_offset_z  # correct for site offset
        target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_open


    elif stage == "retreat":
        target_xy = cube2_pos[:2] + correction_xy
        target_z = cube2_pos[2] + 0.40 - gripper_offset_z  # lift up but compensate for site offset
        target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_open


    else:
        raise ValueError(f"Unknown stage: {stage}")

    # --- Waypoint interpolation ---
    current_pos = eef.get_pos()
    cart_wps = [(1 - alpha) * current_pos + alpha * target_pos for alpha in torch.linspace(0, 1, 8)]

    init_q = robot.get_qpos()
    q_wps = [robot.inverse_kinematics(link=eef, pos=wp, quat=quat, init_qpos=init_q) for wp in cart_wps]

    path = []
    for i in range(len(q_wps) - 1):
        for t in range(10):  # 10 steps between each pair
            alpha = t / 9
            q = (1 - alpha) * q_wps[i] + alpha * q_wps[i + 1]
            path.append(q.clone())

    # --- Gripper interpolation ---
    if stage == "grasp":
        for i in range(len(path) - 5):
            path[i][-1] = grip_open
        for i in range(len(path) - 5, len(path)):
            alpha = (i - (len(path) - 5)) / 5
            path[i][-1] = (1 - alpha) * grip_open + alpha * grip_closed
    else:
        for i in range(len(path)):
            path[i][-1] = grip_val

    return path

def expert_policy_v2(robot, obs, stage):
    """
    Expert policy with extra XY correction before release and retreat.
    """
    eef = robot.get_link("gripper")
    quat = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=gs.device)
    
    from scipy.spatial.transform import Rotation as R
    r = R.from_euler('x', -90, degrees=True)
    quat = torch.tensor(r.as_quat(), dtype=torch.float32)  

    cube1_pos = obs["environment_state"][:3]
    cube2_pos = obs["environment_state"][11:14]

    grip_open = 0.5
    grip_closed = 0.1
    z_offset = 0.18
    correction_xy = torch.tensor([-0.005, 0.02], device=gs.device)
    # Compensate for gripper's site offset in Z
    gripper_offset_z = -0.0981  # from <site name="gripper" ... pos="..." />

    # Correction to align center of gripper with cube2
    gripper_xy_correction = torch.tensor([-0.005, 0.02], device=gs.device)


    if stage == "hover":
        target_pos = cube1_pos + torch.tensor([0.01, 0.02, 0.25], device=gs.device)
        grip_val = grip_open
    elif stage == "wake":
        current_pos = eef.get_pos()
        target_pos = current_pos + torch.tensor([0.0, 0.0, 0.25], device=gs.device)
        grip_val = grip_open
    elif stage == "grasp":
        target_pos = cube1_pos + torch.tensor([0.02, 0.02, 0.045], device=gs.device)
        grip_val = grip_closed

    elif stage == "lift":
        target_pos = cube1_pos + torch.tensor([0.0, 0.0, 0.28], device=gs.device)
        grip_val = grip_closed

    elif stage == "place":
        target_pos = cube2_pos + torch.tensor([0.006, 0.0, z_offset], device=gs.device)
        grip_val = grip_closed

    elif stage == "position_align":
        target_xy = cube2_pos[:2] + gripper_xy_correction
        target_z = cube2_pos[2] + z_offset - gripper_offset_z
        target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_closed


    elif stage == "release":
        target_xy = cube2_pos[:2]  # stay centered
        target_z = cube2_pos[2] + z_offset - gripper_offset_z  # correct for site offset
        target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_open


    elif stage == "retreat":
        target_xy = cube2_pos[:2] + correction_xy
        target_z = cube2_pos[2] + 0.40 - gripper_offset_z  # lift up but compensate for site offset
        target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_open

    elif stage == "go_back":
        # Return to initial position
        target_pos = eef.get_pos()  # We still need a dummy pos for IK
        grip_val = grip_open
        qpos_tensor = torch.deg2rad(torch.tensor([0, -177, 165, 72, -83, 0], dtype=torch.float32, device=gs.device))
        q_wps = [robot.get_qpos(), qpos_tensor]

        path = []
        for i in range(len(q_wps) - 1):
            for t in range(10):  # Interpolate in joint space directly
                alpha = t / 9
                q = (1 - alpha) * q_wps[i] + alpha * q_wps[i + 1]
                q[-1] = grip_val
                path.append(q.clone())
        return path

    else:
        raise ValueError(f"Unknown stage: {stage}")

    # --- Waypoint interpolation ---
    current_pos = eef.get_pos()
    cart_wps = [(1 - alpha) * current_pos + alpha * target_pos for alpha in torch.linspace(0, 1, 8)]

    init_q = robot.get_qpos()
    q_wps = [robot.inverse_kinematics(link=eef, pos=wp, quat=quat, init_qpos=init_q) for wp in cart_wps]

    path = []
    for i in range(len(q_wps) - 1):
        for t in range(10):  # 10 steps between each pair
            alpha = t / 9
            q = (1 - alpha) * q_wps[i] + alpha * q_wps[i + 1]
            path.append(q.clone())

    # --- Gripper interpolation ---
    if stage == "grasp":
        for i in range(len(path) - 5):
            path[i][-1] = grip_open
        for i in range(len(path) - 5, len(path)):
            alpha = (i - (len(path) - 5)) / 5
            path[i][-1] = (1 - alpha) * grip_open + alpha * grip_closed
    else:
        for i in range(len(path)):
            path[i][-1] = grip_val

    return path

def expert_policy_v3(robot, obs, stage):
    """
    Expert policy for placing both red and green blocks onto a tray.
    """
    eef = robot.get_link("gripper")
    from scipy.spatial.transform import Rotation as R
    quat = torch.tensor(R.from_euler('x', -90, degrees=True).as_quat(), dtype=torch.float32, device=gs.device)

    # 各オブジェクトの位置を取得
    red_pos   = obs["environment_state"][:3]
    green_pos = obs["environment_state"][11:14]
    tray_pos  = obs["environment_state"][14:17]

    grip_open = 0.5
    grip_closed = 0.1
    z_offset = 0.18
    correction_xy = torch.tensor([-0.005, 0.02], device=gs.device)
    gripper_offset_z = -0.0981
    gripper_xy_correction = torch.tensor([-0.005, 0.02], device=gs.device)

    # --- 各ステージの制御 ---
    if stage == "hover_red":
        target_pos = red_pos + torch.tensor([0.01, 0.02, 0.25], device=gs.device)
        grip_val = grip_open
    elif stage == "grasp_red":
        target_pos = red_pos + torch.tensor([0.02, 0.02, 0.045], device=gs.device)
        grip_val = grip_closed
    elif stage == "lift_red":
        target_pos = red_pos + torch.tensor([0.0, 0.0, 0.28], device=gs.device)
        grip_val = grip_closed
    elif stage == "place_red":
        target_pos = tray_pos + torch.tensor([-0.04, 0.04, z_offset], device=gs.device)
        grip_val = grip_closed
    # elif stage == "release_red":
    #     target_pos = tray_pos + torch.tensor([-0.03, 0.03, z_offset - gripper_offset_z], device=gs.device)
    #     grip_val = grip_open
    elif stage == "retreat_red":
        target_xy = tray_pos[:2] + torch.tensor([-0.03, 0.03], device=gs.device) + correction_xy
        target_z = tray_pos[2] + 0.40 - gripper_offset_z
        target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_open

    elif stage == "hover_green":
        target_pos = green_pos + torch.tensor([0.01, 0.02, 0.25], device=gs.device)
        grip_val = grip_open
    elif stage == "grasp_green":
        target_pos = green_pos + torch.tensor([0.02, 0.02, 0.045], device=gs.device)
        grip_val = grip_closed
    elif stage == "lift_green":
        target_pos = green_pos + torch.tensor([0.0, 0.0, 0.28], device=gs.device)
        grip_val = grip_closed
    elif stage == "place_green":
        target_pos = tray_pos + torch.tensor([-0.04, 0.04, z_offset], device=gs.device)
        grip_val = grip_closed
    # elif stage == "release_green":
    #     target_pos = tray_pos + torch.tensor([0.04, -0.04, z_offset - gripper_offset_z + 0.005], device=gs.device)
    #     grip_val = grip_open
    elif stage == "retreat_green":
        target_xy = tray_pos[:2] + torch.tensor([-0.03, 0.03], device=gs.device) + correction_xy
        target_z = tray_pos[2] + 0.40 - gripper_offset_z
        target_pos = torch.cat([target_xy, torch.tensor([target_z], device=gs.device)])
        grip_val = grip_open

    elif stage == "go_back":
        target_pos = eef.get_pos()
        grip_val = grip_open
        qpos_tensor = torch.deg2rad(torch.tensor([0, -177, 165, 72, -83, 0], dtype=torch.float32, device=gs.device))
        q_wps = [robot.get_qpos(), qpos_tensor]

        path = []
        for i in range(len(q_wps) - 1):
            for t in range(10):
                alpha = t / 9
                q = (1 - alpha) * q_wps[i] + alpha * q_wps[i + 1]
                q[-1] = grip_val
                path.append(q.clone())
        return path

    else:
        raise ValueError(f"Unknown stage: {stage}")

    # --- Waypoint interpolation ---
    current_pos = eef.get_pos()
    cart_wps = [(1 - alpha) * current_pos + alpha * target_pos for alpha in torch.linspace(0, 1, 8)]
    init_q = robot.get_qpos()
    q_wps = [robot.inverse_kinematics(link=eef, pos=wp, quat=quat, init_qpos=init_q) for wp in cart_wps]

    path = []
    for i in range(len(q_wps) - 1):
        for t in range(10):
            alpha = t / 9
            q = (1 - alpha) * q_wps[i] + alpha * q_wps[i + 1]
            path.append(q.clone())

    # --- Gripper interpolation ---
    if "grasp" in stage:
        for i in range(len(path) - 5):
            path[i][-1] = grip_open
        for i in range(len(path) - 5, len(path)):
            alpha = (i - (len(path) - 5)) / 5
            path[i][-1] = (1 - alpha) * grip_open + alpha * grip_closed
    else:
        for i in range(len(path)):
            path[i][-1] = grip_val

    return path

def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):

        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        # print(observation.keys(), "observation keys")
        # print(observation, "observation")

        for name in observation:
            # print(type(observation[name]), "observation[name] type", name)
            if "image" in name:
                contiguous_array = observation[name].copy()
                tensor = torch.from_numpy(contiguous_array)
                # observation[name] = torch.from_numpy(observation[name])
                observation[name] = tensor.type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        observation["task"] = [task if task else ""]
        observation["robot_type"] = robot_type if robot_type else ""

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action

def remap_observation(obs: dict) -> dict:
    """
    Remaps observation keys from the simulator's format 
    to the format required by the policy.

    Args:
        obs (dict): The original observation dictionary from the environment.
                    Example keys: ['agent_pos', 'pixels']

    Returns:
        dict: The remapped dictionary.
              Example keys: ['observation.state', 'observation.image.top', ...]
    """
    # Create the new dictionary with the desired keys
    remapped_obs = {
        # Assuming 'agent_pos' is the state you want to use
        "observation.state": obs["agent_pos"],
        
        # Map each camera image from the nested 'pixels' dictionary
        "observation.image.top": obs["pixels"]["top"],
        "observation.image.side": obs["pixels"]["side"],
        "observation.image.wrist": obs["pixels"]["wrist"] # Assuming 'wrist' also exists
    }
    return remapped_obs

# === Setup Dataset ===
agent_shape = (8,)
action_shape = (6,)
env_shape = (14,)
dataset_path = Path("data/eval/tray_cube_smolvla_temp")
# dataset_path = Path("data/eval/tray_cube_pi0_v2")

if dataset_path.exists():
    lerobot_dataset = LeRobotDataset(
        repo_id=None,
        root=dataset_path,
    )
else:
    lerobot_dataset = LeRobotDataset.create(
        repo_id=None,
        root=dataset_path,
        robot_type="so101",
        fps=30,
        use_videos=True,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": [
                    "main_shoulder_pan",
                    "main_shoulder_lift",
                    "main_elbow_flex",
                    "main_wrist_flex",
                    "main_wrist_roll",
                    "main_gripper"
                ]
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": [
                    "main_shoulder_pan",
                    "main_shoulder_lift",
                    "main_elbow_flex",
                    "main_wrist_flex",
                    "main_wrist_roll",
                    "main_gripper"
                ]
            },
            "observation.image.top": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.image.side": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.image.wrist": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"]
            },
        },
    )
# no more "wake", 
# stages = [
#     "hover_red", "grasp_red", "lift_red", "place_red", "retreat_red",
# ]
stages = [
    "hover_green", "grasp_green", "lift_green", "place_green", "retreat_green",
]

# === run Episodes ===
eval = True
if eval:
    dataset = LeRobotDataset(
        "iiyudana/tray_cube",
        root=None,
    )

    
    # policy_config = PreTrainedConfig.from_pretrained(model_path) 
    # policy = make_policy(
    #     policy_config,
    #     dataset.meta
    # )
    # policy.to("cuda")
    # model_path = "data/weight/tray_cube_pi0_v2/checkpoints/050000/pretrained_model"
    # policy = PI0Policy.from_pretrained(model_path)

    model_path = "data/weight/tray_cube_smolvla_v2/checkpoints/050000/pretrained_model"
    policy = SmolVLAPolicy.from_pretrained(model_path)
    policy.to("cuda")
    # policy.eval()
    rad2deg = 180 / np.pi
    deg2rad = np.pi / 180
else:
    policy = None

success_count = 0

for ep in range(15):
    if policy is not None:
        policy.reset()
    print(f"\n🎬 Starting episode {ep+1}")
    seed = ep + 1000
    obs, _ = env.reset(seed=seed)
    all_states, all_actions = [], []
    top_frames, side_frames, wrist_frames = [], [], []
    all_rewards = []
    
    if eval:
        for i in range(400):
            print(obs["agent_pos"])
            obs["agent_pos"] = (obs["agent_pos"] * rad2deg)
            obs["agent_pos"][1] *= -1  # flip joint 2
            obs["agent_pos"][4] *= -1  # flip joint 2
            policy_input = remap_observation(obs)
            print(obs["agent_pos"], "obs agent pos")
            action = predict_action(
                policy_input,
                policy,
                device=torch.device("cuda"),
                use_amp=False,
                task="Pick up the green block and place it on the tray.",
                robot_type="so101",
            )
            print(f"Action: {action}")
            # print(f"Action values: {action}")
            action_for_env = action.clone()  # まず、変更可能なコピーを作成
            action_for_env[1] *= -1  # flip joint 2
            action_for_env[4] *= -1  # flip joint 2
            action_for_env = action_for_env * deg2rad
            print(f"Action for env: {action_for_env}")
            obs, reward, done, _, _ = env.step(action_for_env)
            # all_states.append((obs["agent_pos"] * rad2deg).detach().cpu().numpy())
            pos_deg = (obs["agent_pos"] * rad2deg).detach().cpu().numpy()
            pos_deg[1] *= -1  # flip joint 2
            pos_deg[4] *= -1  # flip joint 2
            all_states.append(pos_deg)
            # act_deg = (action * rad2deg).detach().cpu().numpy()
            # act_deg[1] *= -1
            # act_deg[4] *= -1
            all_actions.append(action)
            # all_actions.append((action * rad2deg).detach().cpu().numpy())
            all_rewards.append(reward)

            # Each image is shape (H, W, 3)
            top_frames.append(obs["pixels"]["top"])
            side_frames.append(obs["pixels"]["side"])
            wrist_frames.append(obs["pixels"]["wrist"])
    else:
        for stage in stages:
            action_path = expert_policy_v3(env.get_robot(), obs, stage)
            print(f"Stage: {stage}, action path length: {len(action_path)}")
            for action in action_path:
                obs, reward, done, _, _ = env.step(action)
                # print(stage)

                rad2deg = 180 / np.pi
                # all_states.append((obs["agent_pos"] * rad2deg).detach().cpu().numpy())
                pos_deg = (obs["agent_pos"] * rad2deg).detach().cpu().numpy()
                pos_deg[1] *= -1  # flip joint 2
                pos_deg[4] *= -1  # flip joint 2
                all_states.append(pos_deg)
                act_deg = (action * rad2deg).detach().cpu().numpy()
                act_deg[1] *= -1
                act_deg[4] *= -1
                all_actions.append(act_deg)
                # all_actions.append((action * rad2deg).detach().cpu().numpy())
                all_rewards.append(reward)

                # Each image is shape (H, W, 3)
                top_frames.append(obs["pixels"]["top"])
                side_frames.append(obs["pixels"]["side"])
                wrist_frames.append(obs["pixels"]["wrist"])

                # imageio.imwrite(f"top.png", obs["pixels"]["top"])
                # imageio.imwrite(f"side.png", obs["pixels"]["side"])
                # imageio.imwrite(f"wrist.png", obs["pixels"]["wrist"])
                # breakpoint()
                # imageio.imwrite(f"debug_images/wrist.png", obs["pixels"]["wrist"])


    # Convert to arrays (T, ...)
    states_arr = np.stack(all_states)
    actions_arr = np.stack(all_actions)
    rewards_arr = np.stack(all_rewards)
    top_arr = np.stack(top_frames)
    side_arr = np.stack(side_frames)
    wrist_arr = np.stack(wrist_frames)

    if rewards_arr[-1] > 0:
        success_count += 1
        print(f"🎉 Episode {ep + 1} succeeded! Total successes: {success_count}")
    if True:
        print(f"✅ Saving episode {ep + 1}")
        for t in range(states_arr.shape[0]):
            lerobot_dataset.add_frame({
                "observation.state": states_arr[t],
                "action": actions_arr[t],
                "observation.image.top": top_arr[t],
                "observation.image.side": side_arr[t],
                "observation.image.wrist": wrist_arr[t],
            },
            task="Pick up the green block and place it on the tray.",
            )
        lerobot_dataset.save_episode()
    else:
        print(f"🚫 Skipping episode {ep + 1} — reward was always 0")
print(f"🎉 All episodes succeeded! Total successes: {success_count}")