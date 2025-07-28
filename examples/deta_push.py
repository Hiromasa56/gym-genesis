from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 例：既に作ったデータセット
lerobot_dataset = LeRobotDataset(
    repo_id="iiyudana/tray_cube",
    root="data/train/tray_cube_v2"
)

# Push to hub
lerobot_dataset.push_to_hub(private=False)

# apptainer exec --nv \
#   --bind /run/user/1012 \
#   --bind /var/lib/dbus/machine-id \
#   lerobot_sandbox \
#   bash -c "source /opt/venv/bin/activate && \
#   source /entrypoint.sh && \
#   export PYTHONPATH=$PYTHONPATH:/home/hyamaguchi23/learning_ws/lerobot && \
#   python lerobot/lerobot/scripts/eval.py \
#   --policy.path=outputs/train/aloha_sim_insertion_scripted_image_ep50_50000/checkpoints/020000/pretrained_model \
#   --output_dir=outputs/eval/aloha_sim_insertion_scripted_image_ep50_50000 \
#   --env.type=aloha \
#   --env.task=AlohaInsertion-v0 \
#   --eval.batch_size=10 \
#   --eval.n_episodes=10 \
#   --policy.device=cuda \
#   --policy.use_amp=false"