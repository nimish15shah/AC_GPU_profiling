#!/usr/bin/env python3

import tensorflow as tf
from exec_gpu import exec_gpu
import argparse
import sys

def main(argv=None):
  parser = argparse.ArgumentParser(description='Benchmarking TF+GPU for AC')
  parser.add_argument('ac', type=str, help='Name of AC')
  parser.add_argument('--batch', type=int, default=1, choices=[2**i for i in range(20)], help='Batch size (in multiple of 2)')
  parser.add_argument('--niter', type=int, default= 10, help='Number of iterations')
    
  args = parser.parse_args(argv)

  ac= args.ac
  batch_size= args.batch 
  nIter= args.niter
  
  print('AC:', ac, 'batch_size:', batch_size, 'nIter:', nIter)
  exec_gpu(ac, batch_size, nIter, debug= True)

main()
