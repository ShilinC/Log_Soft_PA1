import numpy as np
import matplotlib.pyplot as plt

from mnist import MNIST

data = MNIST('../python-mnist/data')
images, labels = data.load_training()


