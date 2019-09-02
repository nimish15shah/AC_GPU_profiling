
##  This file converts ac to tensorflow graph
##  It takes as input a pickle file which contains the AC as a dictionary 
##  Each value in the dictionary is node_obj class object from Nimish's graph_analysis project


import tensorflow as tf
import pickle
import networkx as nx
import random
import numpy as np

def load_ac(ac):

  fname= './gr_files/' + ac + '.p'
  with open(fname, 'rb') as fp:
    graph= pickle.load(fp, encoding='latin1')
    fp.close()
  

  fname= './gr_nx_files/' + ac + '_gr.p_nx5ALL_0.00.05.03333_33322_332222_3222222_22222222'
  with open(fname, 'rb') as fp:
    graph_nx= pickle.load(fp)
    
  return graph, graph_nx

def ac_to_tf(ac, batch_size):
  """
    Reads pickled ac, converts it to tf graph
  """
  print('Constructing TF graph from AC')
  graph, graph_nx= load_ac(ac)
  
  #-- Convert ac to tf
  tf_dict= {}
  root= None
  num_ops= 0
  
  print("total node in AC:", graph_nx.number_of_nodes())

  weight_cnt= 0
  ind_cnt= 0

  for node in nx.topological_sort(graph_nx):
#    print(node, end=',')
    obj= graph[node]

    if obj.is_leaf():
      assert len(list(graph_nx.in_edges(node))) == 0
#    if len(list(graph_nx.in_edges(node))) == 0: # Leaf node
#      curr= tf.Variable(tf.random_normal([batch_size,batch_size]), name= 'in')
#      curr= tf.Variable(tf.convert_to_tensor([[[random.random()]*batch_size]*batch_size]), name= 'in')
#      curr= tf.Variable(tf.convert_to_tensor(np.full((batch_size, batch_size), random.random())), name= 'in')
      leaf_type= None
      IND= 0
      WEIGHT= 1
      siblings= set([ch for parent in obj.parent_key_list for ch in graph[parent].child_key_list])
      siblings= siblings - set([node])
      siblings_WEIGHT= False
      siblings_INDICATOR= False

      for sib in siblings:
        if graph[sib].is_weight():
          siblings_WEIGHT= True
        if graph[sib].is_indicator():
          siblings_INDICATOR= True
        
        if siblings_INDICATOR == True and siblings_WEIGHT == True:
          break
        
#      assert not (siblings_WEIGHT == True and siblings_INDICATOR == True)

      if siblings_WEIGHT == True:
        leaf_type= IND
      elif siblings_INDICATOR == True:
        leaf_type= WEIGHT

      if leaf_type== None:
        if len(obj.parent_key_list) == 1:
          leaf_type= WEIGHT
        else:
          leaf_type= IND
      
      if leaf_type == IND:
        ind_cnt += 1
        obj.leaf_type= obj.LEAF_TYPE_INDICATOR
        curr= tf.Variable(tf.convert_to_tensor(np.full((1, batch_size), random.random(), dtype= np.float32)), name= 'ind')
      elif leaf_type== WEIGHT:
        weight_cnt += 1
        obj.leaf_type= obj.LEAF_TYPE_WEIGHT
        curr= tf.constant([random.random()], name= 'weight')
      else:
        assert 0
    
    else: # sum or product
#      assert len(obj.child_key_list) == 2, "AC should be binary"
#      ch_0= tf_dict[obj.child_key_list[0]]
#      ch_1= tf_dict[obj.child_key_list[1]]
#      
#      if obj.operation_type == 1:
#        curr= tf.multiply(ch_0, ch_1, 'mul')
#      elif obj.operation_type == 2:
#        curr= tf.add(ch_0, ch_1, 'mul')
#      else: 
#        assert 0
#      
#      if len(obj.parent_key_list) == 0:
#        assert root== None
#        root= node
#        tf_root= curr
      children= list(graph_nx.predecessors(node))
      parents= list(graph_nx.successors(node))
      
      ch_0= tf_dict[children[0]]
      ch_1= tf_dict[children[1]]

      if random.randint(0,2):
          curr= tf.multiply(ch_0, ch_1, 'mul')
      else:
          curr= tf.add(ch_0, ch_1, 'add')
      
      if len(parents) == 0:
        assert root == None
        root= node
        tf_root= curr

      num_ops += 1

    tf_dict[node]= curr
  
  print("Indicator cnt, Weight Cnt:", ind_cnt, weight_cnt)
  assert root != None
  assert len(tf_dict) == len(graph_nx)

  return tf_root, num_ops
  


