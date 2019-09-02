
import tensorflow as tf
import random

def binary_ac(curr_depth, max_depth= 3, batch_size= 1):
    assert curr_depth <= max_depth

    if curr_depth != max_depth:
        ch_0= binary_ac (curr_depth + 1, max_depth)
        ch_1= binary_ac (curr_depth + 1, max_depth)
        
        if random.randint(0,2):
            curr_op = tf.multiply(ch_0, ch_1, 'mul')
        else:
            curr_op = tf.multiply(ch_0, ch_1, 'add')
#            curr_op = tf.add(ch_0, ch_1, 'add')
        
    elif curr_depth == max_depth:
        curr_op= tf.Variable(tf.random_normal([batch_size, batch_size]), name= 'in')
        
    return curr_op
