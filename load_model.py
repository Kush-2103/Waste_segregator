import tensorflow as tf

# Loading the saved_model
PATH_TO_SAVED_MODEL = "saved_model"

print('Loading model...', end='')

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

print('Done!')
