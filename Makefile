.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: conda # Highly recommended if you want to use your gpu to run the model, in other case you can skip this step and use your cpu instead
conda:
	conda create --name tf python=3.9

.PHONY: conda-install # To install cuda and cudnn packages required for tensorflow use in windows ONLY IF YOU WANT TO USE YOUR GPU
conda-install:
	conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

.PHONY: verify-gpu # Skip this step if you don't have a gpu or you don't want to use it
verify-gpu:
	python face_blur/verify_gpu.py

.PHONY: webcam
webcam:
	python face_blur/webcam.py

.PHONY: image
image:
	python face_blur/image.py