#!/bin/bash
# This script clones the stylegan_xl repository and makes it accessible using the python path.

git_repo="git@github.com:autonomousvision/stylegan_xl.git"
repo_path="./stylegan_xl"
conda activate objdim

if [ ! -d "$repo_path" ]; then
    echo "Cloning repository to ${repo_path}"
    git clone $git_repo $repo_path
    export PYTHONPATH="${PYTHONPATH}:${repo_path}"
    cd $repo_path
    echo "Install requirements from stylegan_xl on top of currently active environment"
    conda env update --file environment.yml --prune
    conda install -c conda-forge ninja # Required by stylegan_xl image generations
else 
    echo "Repository already exists. Skip cloning..."
fi
