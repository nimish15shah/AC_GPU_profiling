import tensorflow as tf
import time
import construct_ac
import numpy as np

# Turn off graph-rewriting optimizations
config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))

# throw error if explicit device placement can't be satisfied
config.allow_soft_placement = False

with tf.device("/gpu:0"):
    # Matrix Multiplication
    N = 4096 
    input1 = tf.Variable(tf.random_normal([N,N]))
    input2 = tf.Variable(tf.random_normal([N,N]))
#    result = tf.matmul(input1, input2)

    # Matrix Add
    N_1 = 4096 
    input1 = tf.Variable(tf.random_normal([N_1,N_1]))
    input2 = tf.Variable(tf.random_normal([N_1,N_1]))
    result= tf.add(input1, input2)

    # Binary tree
    MAX_DEPTH= 7
    BATCH_SIZE= 32
#    result= construct_ac.binary_ac(0, max_depth= MAX_DEPTH, batch_size= BATCH_SIZE) 

    result_no_output = result.op # to avoid transferring data back to Python

sess = tf.Session(config=config)

# load values into GPU
sess.run(tf.global_variables_initializer())

# pre-warming
sess.run(result_no_output)

#num_ops = N**3 + N**2*(N-1)  # N^3 muls, N^2 (N-1) adds
num_ops= (N_1**2) 
#num_ops= (2**(MAX_DEPTH+1)-1) * BATCH_SIZE * BATCH_SIZE

elapsed = []
for i in range(10):
    start = time.time()

    sess.run(result_no_output)
    elapsed.append(time.time()-start)

#print("%d x %d matmul, %.2f elapsed, %.2f G ops/sec"%(N, N, min(elapsed), num_ops/min(elapsed)/10**9))
print("%d x %d matmul, %.2f elapsed, %.2f G ops/sec"%(N, N, np.mean(elapsed), num_ops/np.mean(elapsed)/10**9))
