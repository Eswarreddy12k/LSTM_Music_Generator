import tensorflow as tf
model = tf.saved_model.load('model')
print(model.signatures)

