import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy import linalg as LA

import sys
sys.path.append('../python-mnist/')
from mnist import MNIST

def find_best_parameters(training_images, one_hot_training_labels, test_images, test_labels, validation_images, validation_labels, regularization_types, lambds, weights, inital_step_size, T, epoch, classes, dimensions):

    best_validation_weights_L2 = np.random.randn(dimensions, classes).astype(np.float32) 
    best_validation_lamdb_L2 = 0
    best_validation_accuracy_L2 = 0

    best_validation_weights_L1 = np.random.randn(dimensions, classes).astype(np.float32) 
    best_validation_lamdb_L1 = 0
    best_validation_accuracy_L1 = 0

    early_stoping_threshholds = 5

    for l in regularization_types:
        if l == "L2":

            for lambd in lambds:
                last_accuracy  = 0.0
                cnt = 0
                for i in range(epoch):
                    print i
                    a = np.matmul(training_images, weights) # 18000, 10
                    a_max = np.max(a,1).reshape(a.shape[0],1)
                    sum_exp_a = np.sum(np.exp(a - a_max),1).reshape(a.shape[0],1) # 18000, 1
                    pred_y = np.exp(a - a_max) / (sum_exp_a+0.0) # 18000, 10

                    delta = one_hot_training_labels - pred_y # (18000, 10)
                    grads = -np.dot(training_images.T, delta) + lambd*weights
                    weights = weights - (inital_step_size/(1+i/T)) * grads

                    # Calculate accuracy on validation set
                    a = np.matmul(validation_images, weights) #
                    a_max = np.max(a,1).reshape(a.shape[0],1)
                    sum_exp_a = np.sum(np.exp(a - a_max),1).reshape(a.shape[0],1) # 18000, 1
                    pred_y = np.exp(a - a_max) / (sum_exp_a+0.0) # 18000, 10
                    pred_y[pred_y == 0.0] = 1e-15       
                    pred_class = np.argmax(pred_y, axis=1)
                    validation_accuracy = np.sum(pred_class == validation_labels)/(pred_class.shape[0]+0.0)
                    
                    if validation_accuracy > best_validation_accuracy_L2:
                        best_validation_accuracy_L2 = validation_accuracy
                        best_validation_weights_L2 = weights
                        best_validation_lamdb_L2 = lambd

                    if validation_accuracy < last_accuracy:
                        cnt = cnt + 1
                        if cnt >= early_stoping_threshholds:
                            break
                    else:
                        cnt = 0
                    last_accuracy = validation_accuracy

        elif l == "L1":

            for lambd in lambds:
                last_accuracy  = 0.0
                cnt = 0

                for i in range(epoch):

                    a = np.matmul(training_images, weights)
                    a_max = np.max(a,1).reshape(a.shape[0],1)
                    sum_exp_a = np.sum(np.exp(a - a_max),1).reshape(a.shape[0],1)  
                    pred_y = np.exp(a - a_max) / (sum_exp_a+0.0) 

                    delta = one_hot_training_labels - pred_y # (18000, 10)
                    grads = -np.dot(training_images.T, delta) + lambd* np.sign(weights)
                    weights = weights - inital_step_size/(1+i/T) * grads

                    # Calculate accuracy on validation set
                    a = np.matmul(validation_images, weights) #
                    a_max = np.max(a,1).reshape(a.shape[0],1)
                    sum_exp_a = np.sum(np.exp(a - a_max),1).reshape(a.shape[0],1)
                    pred_y = np.exp(a - a_max) / (sum_exp_a+0.0) 
                    pred_y[pred_y == 0.0] = 1e-15       
                    pred_class = np.argmax(pred_y, axis=1)
                    validation_accuracy = np.sum(pred_class == validation_labels)/(pred_class.shape[0]+0.0)
                    
                    if validation_accuracy > best_validation_accuracy_L1:
                        best_validation_accuracy_L1 = validation_accuracy
                        best_validation_weights_L1 = weights
                        best_validation_lamdb_L1 = lambd

                    if validation_accuracy < last_accuracy:
                        cnt = cnt + 1
                        if cnt >= early_stoping_threshholds:
                            break
                    else:
                        cnt = 0
                    last_accuracy = validation_accuracy

    print best_validation_accuracy_L1
    print best_validation_lamdb_L1
    print best_validation_accuracy_L2
    print best_validation_lamdb_L2

    a = np.matmul(test_images, best_validation_weights_L2) # 18000, 10
    a_max = np.max(a,1).reshape(a.shape[0],1)
    sum_exp_a = np.sum(np.exp(a - a_max),1).reshape(a.shape[0],1) # 18000, 1
    pred_y = np.exp(a - a_max) / (sum_exp_a+0.0) # 18000, 10
    pred_class = np.argmax(pred_y, axis=1)
    test_accuracy = np.sum(pred_class == test_labels)/(pred_class.shape[0]+0.0)
    print "test_accuracy " + str(test_accuracy) 


def plot_graph(training_images, test_images, validation_images, training_labels, test_labels, validation_labels, one_hot_training_labels, one_hot_test_labels, one_hot_validation_labels, lambd, epoch, inital_step_size, T, weights):

    training_losses = []
    test_losses = []
    validation_losses = []

    training_accuracy_ = []
    test_accuracy_ = []
    validation_accuracy_ = []

    for i in range(epoch):
        print "epoch " + str(i)
        a = np.matmul(training_images, weights) 
        a_max = np.max(a,1).reshape(a.shape[0],1)
        sum_exp_a = np.sum(np.exp(a - a_max),1).reshape(a.shape[0],1) 
        pred_y = np.exp(a - a_max) / (sum_exp_a+0.0) 
        delta = one_hot_training_labels - pred_y 
        grads = -np.dot(training_images.T, delta) + lambd*weights
        weights = weights - (inital_step_size/(1+i/T)) * grads

        # Calculate training loss and accuracy
        a = np.matmul(training_images, weights) 
        a_max = np.max(a,1).reshape(a.shape[0],1)
        sum_exp_a = np.sum(np.exp(a - a_max),1).reshape(a.shape[0],1) 
        pred_y = np.exp(a - a_max) / (sum_exp_a+0.0) 
        pred_class = np.argmax(pred_y, axis=1)
        training_accuracy = np.sum(pred_class == training_labels)/(pred_class.shape[0]+0.0)
        training_accuracy_.append(training_accuracy)
        pred_y[pred_y == 0.0] = 1e-15
        log_pred_y = np.log(pred_y)
        training_loss = -np.sum(one_hot_training_labels * log_pred_y)  + lambd*LA.norm(weights)
        training_losses.append(training_loss)
        
        # Calculate test loss and accuracy
        a = np.matmul(test_images, weights) # 18000, 10
        a_max = np.max(a,1).reshape(a.shape[0],1)
        sum_exp_a = np.sum(np.exp(a - a_max),1).reshape(a.shape[0],1) # 18000, 1
        pred_y = np.exp(a - a_max) / (sum_exp_a+0.0) # 18000, 10
        pred_class = np.argmax(pred_y, axis=1)
        test_accuracy = np.sum(pred_class == test_labels)/(pred_class.shape[0]+0.0)
        print test_accuracy
        test_accuracy_.append(test_accuracy)
        pred_y[pred_y == 0.0] = 1e-15
        log_pred_y = np.log(pred_y)
        test_loss = -np.sum(one_hot_test_labels * log_pred_y)  +  lambd*LA.norm(weights)
        test_losses.append(test_loss)
        
        # Calculate validation loss and accuracy
        a = np.matmul(validation_images, weights) # 18000, 10
        a_max = np.max(a,1).reshape(a.shape[0],1)
        sum_exp_a = np.sum(np.exp(a - a_max),1).reshape(a.shape[0],1) # 18000, 1
        pred_y = np.exp(a - a_max) / (sum_exp_a+0.0) # 18000, 10
        pred_class = np.argmax(pred_y, axis=1)
        validation_accuracy = np.sum(pred_class == validation_labels)/(pred_class.shape[0]+0.0) 
        validation_accuracy_.append(validation_accuracy)
        pred_y[pred_y == 0.0] = 1e-15
        log_pred_y = np.log(pred_y)
        validation_loss = -np.sum(one_hot_validation_labels * log_pred_y)  +  lambd*LA.norm(weights)
        validation_losses.append(validation_loss) 


    fig1 = plt.figure(1)
    plt.plot(training_losses)
    plt.plot(test_losses)
    plt.plot(validation_losses)
    fig1.suptitle('Loss VS Epoch', fontsize=15)
    plt.legend(['training loss', 'test loss', 'validation loss'], loc='upper right')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Total loss', fontsize=15)
    fig1.show()

    fig2 = plt.figure(2)
    plt.plot(training_accuracy_)
    plt.plot(test_accuracy_)
    plt.plot(validation_accuracy_)
    fig2.suptitle('Accuracy VS Epoch', fontsize=15)
    plt.legend(['training accuracy', 'test accuracy', 'validation accuracy'], loc='lower right')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    fig2.show()

    # Visualize weights and digits
    weights = weights[1:,:]
    training_images = training_images[:,1:]

    fig3 = plt.figure(figsize = (6,2))
    gs = gridspec.GridSpec(2, classes)
    gs.update(wspace=0, hspace=0)

    for index in range(weights.shape[1]):
        weight = weights[:,index]
        ave_image = np.zeros((1,784))
        for instance in range(training_images.shape[0]):
            if training_labels[instance] == index:
                ave_image += training_images[instance].reshape(1,784)

        ax1 = plt.subplot(gs[0,index])
        plt.imshow(np.reshape(weight, (28,28)), cmap=plt.cm.gray, aspect='equal')
        plt.axis('off')
        ax2 = plt.subplot(gs[1,index])
        plt.imshow(np.reshape(ave_image, (28,28)), cmap=plt.cm.gray, aspect='equal')
        plt.axis('off')
    plt.show()

if __name__ == '__main__':

    data = MNIST('../python-mnist/data')
    training_images, training_labels = data.load_training()
    test_images, test_labels = data.load_testing()

    training_images_old = training_images[:20000]
    training_labels_old = training_labels[:20000]
    test_images = test_images[-2000:]
    test_labels = test_labels[-2000:]
    training_images_old = [[1] + image for image in training_images_old]
    test_images = [[1] + image for image in test_images]

    training_images = np.array(training_images_old[2000:]) / 255.0# (18000, 785)
    training_labels = np.array(training_labels_old[2000:]) 
    validation_images = np.array(training_images_old[:2000]) / 255.0
    validation_labels = np.array(training_labels_old[:2000])
    test_images = np.array(test_images) / 255.0
    test_labels = np.array(test_labels)

    epoch = 300
    inital_step_size = 0.004
    T = 2.0

    lambds_set_1 = [0.01, 0.001, 0.0001]
    lambds_set_2 = [0.05, 0.005, 0.0005]
    lambd = 0.0005
    regularization_types = ["L1", "L2"]

    classes = 10
    dimensions = 785

    weights = np.random.randn(dimensions, classes).astype(np.float32)  #

    one_hot_training_labels = np.eye(classes)[training_labels] 
    one_hot_validation_labels = np.eye(classes)[validation_labels]
    one_hot_test_labels = np.eye(classes)[test_labels]  

    #find_best_parameters(training_images, one_hot_training_labels, test_images, test_labels, validation_images, validation_labels, regularization_types, lambds_set_2, weights, inital_step_size, T, epoch, classes, dimensions)
    plot_graph(training_images, test_images, validation_images, training_labels, test_labels, validation_labels, one_hot_training_labels, one_hot_test_labels, one_hot_validation_labels, lambd, epoch, inital_step_size, T, weights)
