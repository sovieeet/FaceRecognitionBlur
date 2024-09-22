from mtcnn.mtcnn import MTCNN

def MTCNN_setup():

    return MTCNN(
        min_face_size=15, # Minimum size of faces to detect, you can try another values for example 10, 15, 20, reduce this value if the faces are small
        steps_threshold=[0.5, 0.6, 0.6], #Umbrals for each detection stage, you can try another values like 0.6, 0.7, 0.7, make detection more sensitive
        scale_factor=0.7 # Scale factor between image pyramids, you can try another values like 0.5, 0.6, 0.7, more scales to detect faces at different angles/sizes
    )