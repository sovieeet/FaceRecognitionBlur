import tensorflow as tf


def setup_device(
    use_gpu=True,
):  # Change the default value to False to use the CPU instead of GPU
    if use_gpu:
        physical_devices = tf.config.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            print("Using GPU:", physical_devices)
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("No GPU detected, using CPU.")
    else:
        # Force CPU Use
        print("Using CPU only.")
        tf.config.set_visible_devices([], "GPU")
