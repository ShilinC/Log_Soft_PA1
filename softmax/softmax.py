import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
sys.path.append('../python-mnist/')
from mnist import MNIST

def show(image):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()

data = MNIST('../python-mnist/data')
training_images, training_labels = data.load_training()
test_images, test_labels = data.load_testing()

training_images = training_images[:20000]
training_labels = training_labels[:20000]
test_images = test_images[-2000:]
test_labels = test_labels[-2000:]

training_images = [[1] + image for image in training_images]
test_images = [[1] + image for image in test_images]


training_images = np.array(training_images[2000:]) # (18000, 785)
training_labels = np.array(training_labels[2000:])


validation_images = np.array(training_images[:2000])
validation_labels = np.array(training_labels[:2000])

test_images = np.array(test_images)
test_labels = np.array(test_labels)


epoch = 100

step_size = 0.0001
lambd = 0.001

classes = 10
dimensions = 785

weights = np.zeros((dimensions, classes), dtype=float) # 785, 10
one_hot_training_labels = np.eye(classes)[training_labels] # (18000, 10)
one_hot_validation_labels = np.eye(classes)[validation_labels]
one_hot_test_labels = np.eye(classes)[test_labels]  


for i in range(epoch):

    a = np.matmul(training_images, weights) # 18000, 10
    a_max = np.max(a,1).reshape(a.shape[0],1)

    sum_exp_a = np.sum(np.exp(a - a_max),1).reshape(a.shape[0],1) # 18000, 1
    pred_y = np.exp(a - a_max) / (sum_exp_a+0.0) # 18000, 10
    print(pred_y[0])

    delta = one_hot_training_labels - pred_y # (18000, 10)
    weights = weights + step_size * np.dot(training_images.T, delta) 

    '''
    # Calculate loss
    pred_y[pred_y == 0.0] = 1e-10
    log_pred_y = np.log(pred_y)
    cross_entropy = -np.sum(one_hot_training_labels * log_pred_y) / training_images.shape[0]
    print cross_entropy
    '''

    # Calculate loss on validation set

    a = np.matmul(validation_images, weights) #
    a_max = np.max(a,1).reshape(a.shape[0],1)
    sum_exp_a = np.sum(np.exp(a - a_max),1).reshape(a.shape[0],1) # 18000, 1
    pred_y = np.exp(a - a_max) / (sum_exp_a+0.0) # 18000, 10
    pred_y[pred_y == 0.0] = 1e-15   
    log_pred_y = np.log(pred_y)
    cross_entropy = -np.sum(one_hot_validation_labels * log_pred_y) / validation_images.shape[0]
    print cross_entropy

    # Calculate accuracy
    pred_class = np.argmax(pred_y, axis=1)

    #print(np.sum(pred_class == training_labels)/(pred_class.shape[0]+0.0))


   





