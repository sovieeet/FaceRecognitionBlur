# Face Recognition

This repositor it's a personal face recognition and blur project. This project uses the libraries OpenCV, Tensorflow and MTCNN to recognize faces and blur it.

## Setup

1. **Clone the Repository**:
     ```sh
    git clone https://github.com/sovieeet/FaceRecognitionBlur.git
    cd face_recognition
     ```

2. **Use conda env**:
    This is highly recommended if you want to use your gpu to run the model
    ```sh
    conda create --name tf python=3.9
     ```
    Or if you have installed Make (Unix systems brings this by default)
     ```sh
    make conda
     ```

3. **Or simply use venv**:
    For windows users:
    ```sh
    python -m venv .venv
     ```
    For Mac/Linux/WSL users:
    ```sh
    python3 -m venv .venv
     ```
    Or if you have installed Make (for Windows users)
     ```sh
    make venv-win
     ```
    Or if you have installed Make (for Mac/Linux users)
    ```sh
    make venv-unix
     ```
    Activate the venv for Windows users:
    ```sh
    make venv-win-act
     ```
    Activate the venv for Linux/Mac users:
    ```sh
    make venv-unix-act
     ```

3. **Install CUDA and cuDNN for GPU Use**:
    If you want to your GPU to run the model, you must need CUDA and cuDNN libraries instaling by the next way:
    ```sh
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
     ```
    Or if you have installed Make
     ```sh
    make conda-install
     ```

3. **Install Dependencies**:
     ```sh
    pip install -r requirements.txt
     ```

4. **Run the Application**:

    - **First, go to `face_blur/face_processing.py`** and in the case you don't want to use a GPU, change `use_gpu=True` to `use_gpu=False`:
    ```python
    def setup_device(use_gpu=True)
     ```

    - **If you want to recognize faces and blur it with images (THIS ONLY WORKS FOR WINDOWS USERS, PENDING DOC TO MAC/LINUX USERS)**:
     ```sh
    python face_blur/image.py
     ```
     or
     ```sh
    make image
     ```

    - **If you want to recognize faces and blur it with webcam (THIS ONLY WORKS FOR WINDOWS USERS, PENDING DOC TO MAC/LINUX USERS)**:
     ```sh
    python face_blur/webcam.py
     ```
     or
     ```sh
    make webcam
     ```

## Usage

- **Recognize Faces using images**: Run the `face_blur/image.py` script to recognize faces using an image in a path.
- **Recognize Faces using webcam**: Run the `face_blur/webcam.py` script to recognize faces in real-time.

## Recommendation

You can play with the params in `face_processing.py`, `blur_processing.py` and `MTCNN_settings.py` to compare the results with different images.# FaceRecognitionBlur
