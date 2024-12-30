init: install

install:
	pip install -U -e sohoo
	pip install -U -e shared

run-sohoo:
	python -m sohoo
