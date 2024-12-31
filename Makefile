init: install

install:
	pip install -U -e sohoo
	pip install -U -e shared
	pip install -U -e models

	cp models/src/neural_network/trainers_model.pth model_serve/src/neural_api/weight_model.pth
	cp models/src/neural_network/class_names.json model_serve/src/neural_api/
	pip install -U -e model_serve

install-support-m1:
	pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
	pip install numpy pandas scikit-learn matplotlib jupyter

run-sohoo:
	python -m sohoo

run-training-neural:
	python -m neural_network

run-neuralapi:
	python -m neural_api
