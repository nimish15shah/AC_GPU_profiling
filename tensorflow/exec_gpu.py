
import tensorflow as tf
import time
from ac_to_tf import ac_to_tf
import numpy as np

def exec_gpu(ac, batch_size, nIter, debug= False):
  # Turn off graph-rewriting optimizations
  config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
  
  # throw error if explicit device placement can't be satisfied
  config.allow_soft_placement = False
  
  with tf.device("/gpu:0"):
    root, n_ac_ops=  ac_to_tf(ac, batch_size)
    result_no_output= root.op

  sess = tf.Session(config=config)
  
  # load values into GPU
  sess.run(tf.global_variables_initializer())
  
  # pre-warming
  sess.run(result_no_output)
  
  if debug:
    # Profiler is created here.
    profiler = tf.profiler.Profiler(sess.graph)
    options = tf.RunOptions()
    options.trace_level = tf.RunOptions.FULL_TRACE
#    options.trace_level = tf.RunOptions.SOFTWARE_TRACE
#    options.trace_level = tf.RunOptions.HARDWARE_TRACE
    metadata = tf.RunMetadata()

  

  elapsed = []
  N= nIter
  for i in range(N):
    start = time.time()
    if debug:
      sess.run(result_no_output, options= options, run_metadata=metadata) 
      profiler.add_step(i, metadata)
    else:
      sess.run(result_no_output)
    elapsed.append(time.time()-start)
  
  avg_time= np.mean(elapsed)
  num_ops= n_ac_ops * batch_size
  through= num_ops/avg_time/(10**6)
  freq= 1.12 * (10**9)
  ops_cycle= num_ops/(avg_time*freq)
  
  print("%d num_ops, %.2f elapsed, %.6f M ops/sec, %.4f ops/cycle"%(num_ops, avg_time, through , ops_cycle))

  if debug:
    writer = tf.summary.FileWriter("./tmp/log/", sess.graph)
    writer.close()

    from tensorflow.python.client import timeline
    # create the Timeline object, and write it to a json file
    fetched_timeline = timeline.Timeline(metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('./timeline_'+ ac +'.json', 'w+') as f:
        f.write(chrome_trace)

    option_builder = tf.profiler.ProfileOptionBuilder
    opts = (option_builder(option_builder.time_and_memory()).
            with_step(-1). # with -1, should compute the average of all registered steps.
            with_file_output('test-%s.txt' % N).
            select(['micros','bytes','occurrence']).order_by('micros').
            build())
    # Profiling infos about ops are saved in 'test-%s.txt' % FLAGS.out
#    profiler.profile_operations(options=opts)

#    writer.add_run_metadata(metadata, 'step%d' % i)
