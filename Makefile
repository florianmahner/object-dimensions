.PHONY: install
install:
	echo "Creating conda environment 'deep' from environment.yaml"
	conda env create -f environment.yaml python=3.9
	conda activate deep
	pip install torch torchvision --upgrade 
	pip install -e .