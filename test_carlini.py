## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

from attacks.l2_attack import CarliniL2
from attacks.l0_attack import CarliniL0
from attacks.li_attack import CarliniLi


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 100, 'How many train steps')
tf.app.flags.DEFINE_string('dataset', '', 'How many train steps')

def perform_attack(attack, data, labels, sess):
    nb_sample = data.shape[0]
    batch_size = FLAGS.batch_size 
    nb_batch = int(np.ceil(nb_sample/batch_size))
    X_adv = np.empty(data.shape)
    for batch in range(nb_batch):
        print('batch {0}/{1}'.format(batch+1, nb_batch), end='\r')
        start = batch * batch_size
        end = min(nb_sample, start+batch_size)
        inputs = data[start:end]
        targets = labels[start:end]
        tmp = attack.attack(inputs, targets)
        X_adv[start:end] = tmp
    return X_adv

def main(_):
    with tf.Session() as sess:
        if FLAGS.dataset == 'MNIST':
            data, model =  MNIST(), MNISTModel("models/mnist", sess)
        elif FLAGS.dataset == 'Cifar':
            data, model =  CIFAR(), CIFARModel("models/cifar", sess)

        attack = CarliniL2(sess, model, batch_size=FLAGS.batch_size, max_iterations=1000, confidence=0)
        train_data = data.train_data
        train_labels = data.train_labels
        test_data = data.test_data
        test_labels = data.test_labels

        X_adv_test = perform_attack(attack, test_data, test_labels, sess)
        X_adv_train = perform_attack(attack, train_data, train_labels, sess)

        np.save('adversarial_outputs/carlini_train_' + FLAGS.dataset.lower() + '.npy', X_adv_train)
        np.save('adversarial_outputs/carlini_test_' + FLAGS.dataset.lower() + '.npy', X_adv_test)
        print("Legit/Adversarial training set")
        model.evaluate(train_data, train_labels)
        model.evaluate(X_adv_train, train_labels)
        
        print("Legit/Adversarial test set")
        model.evaluate(test_data, test_labels)
        model.evaluate(X_adv_test, test_labels)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
