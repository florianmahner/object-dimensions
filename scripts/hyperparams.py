import subprocess
import numpy as np
import os

# Behavior
# BETAS = np.arange(0.15, 0.9, 0.05).round(2)
# SEEDS = range(2)

# Deep 256 Batch size 
# BETAS = list(np.arange(0.1, 1.05, 0.05).round(2))
# SEEDS = range(1)

# Reproducbility seed comparison!
BETAS = [0.55]
SEEDS = range(30)

N_RESUB = 4


for seed in SEEDS:
    for beta in BETAS:

        with open("./scripts/slurm/slurm_script.txt") as f:
            slurm = f.readlines()

        # cmd = '\nsrun python3 ~/deep_embeddings/deep_embeddings/main.py --config "./configs/train_deep.toml" --beta {} --seed {} --init_dim 200 --batch_size 256 --load_model'.format(beta, seed)
        # cmd = '\nsrun python3 ~/deep_embeddings/deep_embeddings/main.py --config "./configs/train_deep.toml" --beta {} --seed {} --init_dim 500 --batch_size 256 --load_model'.format(beta, seed)
        cmd = '\nsrun python3 ~/deep_embeddings/deep_embeddings/main.py --config "./configs/train_deep.toml" --beta {} --seed {} --init_dim 500 --batch_size 16384 --identifier "reproducibility" --stability_time 1000 --load_model'.format(beta, seed)
        # cmd = '\nsrun python3 ~/deep_embeddings/deep_embeddings/main.py --config "./configs/train_behavior.toml" --beta {} --seed {} --init_dim 500 --load_model'.format(beta, seed)

        slurm.append(cmd)

        slurm_fn = './scripts/slurm/beta_{}_seed_{}_slurm.sh'.format(beta, seed)
        with open(slurm_fn, 'w') as f:
            f.writelines(slurm)
 
        # Call the bash script and get the job id
        job_id = subprocess.check_output(['sbatch',  '{}'.format(slurm_fn)])
        job_id = job_id.decode('utf-8').split()[-1]

        # Submit the walltime limit, by sumitting the job n times
        # for i in range(N_RESUB):
        #     subprocess.call(['sbatch', '--dependency=afterany:{}'.format(job_id), '{}'.format(slurm_fn)])
        # subprocess.call(['sbatch',  '{}'.format(slurm_fn)])        

        # Remove the slurm script
        os.remove(slurm_fn)

