.PHONY: data stylegan

data:
	@python scripts/get_things_data.py

stylegan:
	@bash scripts/get_stylegan_xl.sh