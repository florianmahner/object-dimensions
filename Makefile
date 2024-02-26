.PHONY: data
.PHONY: stylegan

data:
    bash scripts/get_things_data.sh

stylegan:
	python scripts/get_stylegan_xl.sh