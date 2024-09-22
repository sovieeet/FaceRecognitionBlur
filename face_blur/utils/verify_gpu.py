import tensorflow as tf
"""
ONLY USE THIS IF YOU PLAIN TO USE GPU TO TEST IF IT IS WORKING WITH TENSORFLOW
"""
print("Tensorflow version:", tf.__version__)

physical_devices = tf.config.list_physical_devices('GPU')
print("GPU devices availables:", physical_devices)
