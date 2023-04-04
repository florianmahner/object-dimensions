.PHONY: install
install:
	echo "Creating conda environment 'deep' from environment.yaml"
	conda env create -f environment.yaml python=3.9
	@echo "Activate the conda environment by running 'conda activate deep'"
	@echo "Install dependencies by running 'pip install -e .'"