import os

from imitation.data import serialize, huggingface_utils, types


def safe_save_dagger_demo(traj: types.Trajectory, traj_index, save_dir, rng):
    # Ensure the index is within the dataset size
    dataset_size = len(traj)
    if traj_index >= dataset_size:
        print(f"Skipping save: traj_index={traj_index} exceeds dataset_size={dataset_size}")
        return

    # Save trajectory
    npz_path = os.path.join(save_dir, f"trajectory_{traj_index}.npz")
    safe_save(npz_path, [traj])
    print(f"Saved trajectory shard to {npz_path}")


def safe_save(npz_path, trajectories: list[types.Trajectory]):
    print(f"Saving trajectories: {len(trajectories)} trajectories provided.")
    for i, traj in enumerate(trajectories):
        print(f"Trajectory {i}: obs={len(traj.obs)}, actions={len(traj.acts)}")

    # Convert trajectories to dataset
    dataset = huggingface_utils.trajectories_to_dataset(trajectories)
    dataset_size = len(dataset)
    print(f"Converted dataset size: {dataset_size}")

    # Ensure the number of shards is at most the dataset size
    num_shards = min(dataset_size, len(trajectories))
    print(f"Saving dataset with {num_shards} shard(s)...")

    if dataset_size == 0:
        print("Dataset is empty. No shards to save.")
        return

    if num_shards == 1:
        # Save the entire dataset as a single shard
        shard_path = f"{npz_path}_shard_0"
        dataset.save_to_disk(shard_path, num_shards=num_shards)
        print(f"Single shard saved to {shard_path}")
        return

    # Save multiple shards if applicable
    for shard_idx in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=shard_idx, contiguous=True)
        shard_path = f"{npz_path}_shard_{shard_idx}"
        shard.save_to_disk(shard_path)
        print(f"Shard {shard_idx} saved to {shard_path}")
