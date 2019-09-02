#!/usr/bin/python3

import tensorflow as tf
import random
import math
import time

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2

import construct_ac

print(tf.__file__)

MAX_DEPTH= 7
root= construct_ac.binary_ac(0, max_depth= MAX_DEPTH) 

BATCH_SIZE= (2**8)-1
#A= tf.Variable([1.2]*BATCH_SIZE)
A= tf.Variable([[1.3]*BATCH_SIZE]*BATCH_SIZE)
B= tf.Variable([[1.3]*BATCH_SIZE]*BATCH_SIZE)
C= tf.multiply(A, B)
C= tf.matmul(A, B)
root= C

# to avoid writing resultback to python 
result_no_output= root.op

# add an Op to initialize global variables.
init_op = tf.global_variables_initializer()


def no_rewrite_session_config():
  rewriter_config = rewriter_config_pb2.RewriterConfig(
      disable_model_pruning=True,
      constant_folding=rewriter_config_pb2.RewriterConfig.OFF)
  graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
  return config_pb2.ConfigProto(graph_options=graph_options)


# Turn off graph-rewriting optimizations
config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))

# throw error if explicit device placement can't be satisfied
config.allow_soft_placement = False

tf.device('/gpu:0')
#with tf.Session(config=no_rewrite_session_config()) as sess:
with tf.Session(config=config) as sess:

    # run the Op that initializes global variables.
    sess.run(init_op)

    options = tf.RunOptions()
#    options.trace_level = tf.RunOptions.FULL_TRACE
#    options.trace_level = tf.RunOptions.SOFTWARE_TRACE
    metadata = tf.RunMetadata()

    #warmup
    sess.run(result_no_output)

    print('Running')
    start= time.time()
    N= 100
    for i in range(N):
#        marginal_val_arr = sess.run(op6, options= options, run_metadata=metadata) 
        marginal_val_arr = sess.run(result_no_output) 

    time_taken = time.time()- start

    print('Time_taken/Inference:', time_taken/N)    
    print('# of nodes', 2**(MAX_DEPTH+1)-1)
    print(marginal_val_arr)
    
    # print the timings of each operation that executed.
#    print(metadata.step_stats)

#    writer = tf.summary.FileWriter("./tmp/log/", sess.graph)
#    writer.add_run_metadata(metadata, 'step%d' % i)
#    writer.close()
    
    from tensorflow.python.client import timeline
    # create the Timeline object, and write it to a json file
    fetched_timeline = timeline.Timeline(metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('./timeline_01.json', 'w+') as f:
        f.write(chrome_trace)


exit(1)

######################################333
import libspn as spn
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

tf.ConfigProto(log_device_placement=True)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#print(sess.run(c))

#with tf.Session() as sess:
#    print (sess.run(c))


iv_x = spn.IndicatorLeaf(num_vars=2, num_vals=2, name="iv_x")
gen=spn.DenseSPNGenerator(num_decomps=1, num_subsets=2, num_mixtures=2)
root=gen.generate(iv_x, root_name="root")
iv_y = root.generate_latent_indicators(name="iv_y") # Can be added manually
spn.generate_weights(root, initializer=tf.initializers.random_uniform(0, 1)) # Can be added manually

print(root.get_num_nodes())
print(root.get_scope())
print(root.is_valid())

SUM_CNT= 8 
sum_ls= []
#iv_x = spn.IndicatorLeaf([[0,-1], [-1,-1]] ,num_vars=2, num_vals=2, name="iv_x")
for i in range(SUM_CNT):
    iv_x = spn.IndicatorLeaf( [[-1],[0]] , num_vars=1, num_vals=2, name="iv_x" + str(i))
#for i in range(2):
    sum_x = spn.Sum((iv_x, [0,1]), name="sum_" + str(i))
    sum_x.generate_weights(tf.initializers.constant([random.random(), random.random()]))
    sum_ls.append(sum_x)
#for i in range(2,4):
#    sum_x = spn.Sum((iv_x, [0,1]), name="sum_" + str(i))
#    sum_x.generate_weights(tf.initializers.constant([random.random(), random.random()]))
#    sum_ls.append(sum_x)

LAST_CNT=SUM_CNT
last_node_ls= sum_ls
node_type= 'Sum'
LAYER_CNT= int(math.log(SUM_CNT/2, 2))
print('Layer CNT', LAYER_CNT)
for l in range(LAYER_CNT):
#    CURR_CNT= int(3*LAST_CNT/4)
    CURR_CNT= int(LAST_CNT/2)
    curr_node_ls= []
    if node_type== 'Sum':
        node_type= 'Prod'
    else:
        node_type= 'Sum'
    
    node_type= 'Prod'

    for i in range(CURR_CNT):
#        ch_0= random.choice(last_node_ls)
#        ch_1= random.choice(last_node_ls)
        ch_0= last_node_ls[i*2]
        ch_1= last_node_ls[i*2+1]
        if node_type== 'Prod':
            node_x= spn.Product( ch_0, ch_1, name= "prod" + str(l) + '_' + str(i))
        elif node_type== 'Sum':
            iv_x = spn.IndicatorLeaf(num_vars=2, num_vals=2, name="iv_x" + str(l) + '_'+ str(i))
            node_x= spn.Sum( ch_0, ch_1, name= "sum" + str(l) + '_' + str(i))
            node_x.generate_weights(tf.initializers.constant([random.random(), random.random()]))
        else:
            assert 0
        curr_node_ls.append(node_x)

    last_node_ls= curr_node_ls 
    LAST_CNT= CURR_CNT

root = spn.Sum(last_node_ls[0], last_node_ls[1], name="root")
root.generate_weights(tf.initializers.constant([0.5, 0.2]))
iv_y = root.generate_latent_indicators(name="iv_y")

print('num_of_nodes', root.get_num_nodes())
#print('scope', root.get_scope())
#print('is_valid', root.is_valid())

init_weights = spn.initialize_weights(root)
marginal_val = root.get_value(inference_type=spn.InferenceType.MARGINAL)
#iv_x_arr = [[0, 1],
#           [0, -1],
#           [-1,-1]]

iv_y_arr = [[0], [-1]]


tf.device('/gpu:0')

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./tmp/log/", sess.graph)
    options = tf.RunOptions()
    options.trace_level = tf.RunOptions.FULL_TRACE
    metadata = tf.RunMetadata()

    init_weights.run()
    print('Running')
    start= time.time()
    marginal_val_arr = sess.run(marginal_val, feed_dict={ iv_y: iv_y_arr}, options= options, run_metadata=metadata) #iv_x: iv_x_arr,
    print('Time_taken:', time.time()-start)    

    print(marginal_val_arr)
    
    # Print the timings of each operation that executed.
#    print(metadata.step_stats)
    writer.close()
    
    from tensorflow.python.client import timeline
    # Create the Timeline object, and write it to a json file
    fetched_timeline = timeline.Timeline(metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('./timeline_01.json', 'w+') as f:
        f.write(chrome_trace)
#spn.display_spn_graph(root)

