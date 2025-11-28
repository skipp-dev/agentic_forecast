import tensorflow as tf
print("tensorflow", tf.__version__)
print("GPUs:", len(tf.config.list_physical_devices("GPU")))
