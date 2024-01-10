import tensorflow as tf

tensor_8d = tf.constant(42)

print(tensor_8d.numpy())

tensor_1d = tf.constant([1,2,3,4])
print(tensor_1d.numpy())

tensor_2d = tf.constant([[1,2,3],[4,5,6]])
print(tensor_2d)


tensor_3d = tf.constant([[[1, 2], [3, 4], [5, 6], [7, 8]]])
print(tensor_3d)
