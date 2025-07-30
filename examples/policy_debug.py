import os
import argparse
import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import matplotlib.pyplot as plt

class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self):
        return len(self.frame_ids)

def unsqueeze_if_needed(x):
    return x if x.ndim == 4 else x.unsqueeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, default=None, help='Path to dataset root')
    parser.add_argument('--repo-id', type=str, required=True, help='Dataset repo_id')
    parser.add_argument('--policy-path', type=str, required=True, help='Path to trained policy directory')
    parser.add_argument('--episode-index', type=int, default=0, help='Episode index to evaluate')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs("data/plots", exist_ok=True)
    save_path = f"data/plots/episode_{args.episode_index}_actions.png"

    # Load the dataset
    print(f"Loading dataset from {args.dataset_root} with repo_id {args.repo_id}...")
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    print(f"Loaded dataset: {len(dataset)} samples, {dataset.num_episodes} episodes")

    # Sample one episode
    sampler = EpisodeSampler(dataset, args.episode_index)

    # Load the policy
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    # policy = PI0Policy.from_pretrained(args.policy_path)  # Uncomment if using PI0Policy
    policy.to(args.device)
    # policy.eval()

    # Evaluate the policy
    errors = []
    pred_actions = []
    gt_actions = []
    for i in sampler:
        sample = dataset[i]

        obs = {
            "observation.state": sample["observation.state"].unsqueeze(0).to(args.device),
            "observation.image.top": unsqueeze_if_needed(sample["observation.image.top"]).to(args.device),
            "observation.image.side": unsqueeze_if_needed(sample["observation.image.side"]).to(args.device),
            "observation.image.wrist": unsqueeze_if_needed(sample["observation.image.wrist"]).to(args.device),
            "task": [sample["task"]],
        }
        # print(obs["observation.images.cam_high"].shape)
        gt_action = torch.tensor(sample["action"], dtype=torch.float32).to(args.device)

        with torch.no_grad():
            pred_action = policy.select_action(obs).squeeze(0)

        threshold = 10.0
        for j in range(len(pred_action)):
            diff = torch.abs(pred_action[j] - gt_action[j])
            if diff >= threshold:
                print(f"frame{i}, Index {j}: pred = {pred_action[j]:.4f}, gt = {gt_action[j]:.4f}, diff = {diff:.4f}")

        error = torch.norm(pred_action - gt_action).item()
        errors.append(error)
        pred_actions.append(pred_action.cpu().numpy())
        gt_actions.append(gt_action.cpu().numpy())

    print(f"\nMean L2 error for episode {args.episode_index} over {len(errors)} frames: {sum(errors)/len(errors):.4f}")

    # Plotting
    pred_actions = torch.tensor(pred_actions)
    gt_actions = torch.tensor(gt_actions)
    num_joints = pred_actions.shape[1] if pred_actions.ndim > 1 else 1
    frames = range(len(pred_actions))
    plt.figure(figsize=(10, 6))
    for j in range(num_joints):
        plt.plot(frames, gt_actions[:, j], label=f"GT action {j}", linestyle='--')
        plt.plot(frames, pred_actions[:, j], label=f"Pred action {j}")
    plt.xlabel("Frame")
    plt.ylabel("Action Value")
    plt.title(f"GT vs Predicted Actions for Episode {args.episode_index}")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path)  # ← 保存
    plt.close()  # リソースを解放（重要）

    print(f"アクション比較プロットを保存しました: {save_path}")

if __name__ == "__main__":
    main()