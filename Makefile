.PHONY: images data stylegan

data:
	@echo "Downloading data from OSD..."
	@wget https://osf.io/download/vuqg8/ -O data.tar.gz
	@echo "Extracting data..."
	@tar -xvf data.tar.gz
	@echo "Data download and extraction complete."

images:
	@echo "Downloading images from OSF..."
	@python scripts/get_things_data.py
	@echo "Images download complete."

stylegan:
	@echo "Downloading StyleGAN XL..."
	@bash scripts/get_stylegan_xl.sh
	@echo "StyleGAN XL download complete."