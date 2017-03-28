import logging
import os
import sys


# noinspection PyAttributeOutsideInit
class AverageMeter(object):
  """
  Computes and stores the average and current value
  """
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val*n
    self.count += n
    self.avg = self.sum/self.count


def get_logger(path):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

  fh = logging.FileHandler(os.path.join(path, 'debug.log'))
  fh.setLevel(logging.INFO)
  fh.setFormatter(formatter)
  logger.addHandler(fh)

  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)
  ch.setFormatter(formatter)
  logger.addHandler(ch)

  return logger