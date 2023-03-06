import subprocess
import numpy as np
import os


# Deep 256 Batch size
# BETAS = list(np.arange(0.1, 1.05, 0.05).round(2))


# Reproducbility seed comparison!
# BETAS = [0.55]
# SEEDS = range(30)
# SEEDS = [0]


# Uing this right now!
# Batch size 256 300 dims
# BETAS = np.arange(0.1, 0.34, 0.02).round(2)
BETAS = np.arange(0.16, 0.34, 0.02).round(2)
SEEDS = range(4)


# BETAS = [0.18, 0.22, 0.26]
# SEEDS = [1]

# BETAS = [0.24]
# SEEDS = [0,1]

# Using this for 300 dims Behavior VICE
# BETAS = list(np.arange(0.6, 0.9, 0.05).round(2))
# SEEDS = range(2)

# Using this for finegrained 300 dims Behavior VICE
# BETAS = list(np.arange(0.75, 0.95, 0.01).round(2))
# SEEDS = range(5)


# Fine grained spose behavior
# BETAS = np.arange(0.0031, 0.0043, 0.0001).round(4)
# SEEDS = range(5)

# BETAS = np.arange(0.92, 0.96, 0.01).round(2)
# SEEDS = range(1,4)

# SPOSE martin recon
# BETAS = [0.00385]
# SEEDS = range(2)
# SEEDS = [0, 1]


# CHECK OUT SPOSE BRANCH!!!

N_RESUB = 2
for seed in SEEDS:
    for beta in BETAS:
        with open("./scripts/slurm/slurm_script.txt") as f:
            slurm = f.readlines()

        cmd = '\nsrun python3 ~/deep_embeddings/deep_embeddings/main.py --config "./configs/train_256_bs.toml" --beta {} --seed {} --load_model'.format(
            beta, seed
        )

        # cmd = '\nsrun python3 ~/deep_embeddings/deep_embeddings/main.py --config "./configs/train_behavior.toml" --beta {} --seed {} --batch_size 256 --init_dim 300 --load_model'.format(beta, seed)

        # cmd = '\nsrun python3 ~/deep_embeddings/deep_embeddings/main.py --config "./configs/train_spose_behavior.toml" --beta {} --seed {} --init_dim 300 --batch_size 256 --method "deterministic" --load_model'.format(beta, seed)

        # cmd = '\nsrun python3 ~/deep_embeddings/deep_embeddings/main.py --config "./configs/train_spose_behavior.toml" --beta {} --seed {} --init_dim 90 --method "deterministic" --load_model'.format(beta, seed)

        # cmd = '\nsrun python3 ~/deep_embeddings/deep_embeddings/main.py --config "./configs/train_deep.toml" --beta {} --seed {} --init_dim 200 --batch_size 256 --load_model'.format(beta, seed)
        # cmd = '\nsrun python3 ~/deep_embeddings/deep_embeddings/main.py --config "./configs/train_deep.toml" --beta {} --seed {} --init_dim 500 --batch_size 256 --load_model'.format(beta, seed)
        # cmd = '\nsrun python3 ~/deep_embeddings/deep_embeddings/main.py --config "./configs/train_deep.toml" --beta {} --seed {} --init_dim 300 --batch_size 256 --load_model'.format(beta, seed)
        # cmd = '\nsrun python3 ~/deep_embeddings/deep_embeddings/main.py --config "./configs/train_deep.toml" --beta {} --seed {} --init_dim 300 --batch_size 16384 --identifier "reproducibility" --stability_time 1000 --load_model'.format(beta, seed)

        # cmd = "srun --time=00:05:00 --nodes=1 --tasks-per-node=1 --cpus-per-task=40 --partition=gpudev --gres=gpu:v100:2 "

        slurm.append(cmd)

        slurm_fn = "./scripts/slurm/beta_{}_seed_{}_slurm.sh".format(beta, seed)
        with open(slurm_fn, "w") as f:
            f.writelines(slurm)

        # Call the bash script and get the job id
        job_id = subprocess.check_output(["sbatch", "{}".format(slurm_fn)])
        job_id = job_id.decode("utf-8").split()[-1]

        # Submit the walltime limit, by sumitting the job n times
        for i in range(N_RESUB):
            job_id = subprocess.check_output(
                [
                    "sbatch",
                    "--dependency=afterany:{}".format(job_id),
                    "{}".format(slurm_fn),
                ]
            )
            job_id = job_id.decode("utf-8").split()[-1]

        # Remove the slurm script
        os.remove(slurm_fn)
