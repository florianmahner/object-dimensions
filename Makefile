.PHONY: install
install:
	echo "Creating conda environment 'objdim' from environment.yaml"
	conda env create -f environment.yaml python=3.9
	@echo "Install StyleGAN XL dependencies"
	source scripts/get_stylegan_xl.sh
	@echo "Activate the conda environment by running 'conda activate objdim'"
	@echo "Install dependencies by running 'pip install -e .'"