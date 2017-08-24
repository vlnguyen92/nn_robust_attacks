## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time

from keras import backend as K

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

from attacks.fgsm import fgsm

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 100, 'How many train steps')
tf.app.flags.DEFINE_string('dataset', '', required=True, 'How many train steps')

def attack(x_adv, data, labels, sess):
    nb_sample = data.shape[0]
    batch_size = 100
    nb_batch = int(np.ceil(nb_sample/batch_size))
    X_adv = np.empty(data.shape)
    for batch in range(nb_batch):
        print('batch {0}/{1}'.format(batch+1, nb_batch), end='\r')
        start = batch * batch_size
        end = min(nb_sample, start+batch_size)
        tmp = sess.run(x_adv, feed_dict={x: data[start:end],
                                         y: labels[start:end],
                                         K.learning_phase(): 0})
        X_adv[start:end] = tmp
    return X_adv

if FLAGS.dataset == 'MNIST':
    x = tf.placeholder(tf.float32, (None, 28, 28, 1))
elif FLAGS.datset == 'Cifar':
    x = tf.placeholder(tf.float32, (None, 32, 32, 3))

y = tf.placeholder(tf.float32, (None, 10))

def main(_):
    with tf.Session() as sess:
        K.set_session(sess)
        if FLAGS.dataset == 'MNIST':
            data, model =  MNIST(), MNISTModel("models/mnist", sess)
        elif FLAGS.datset == 'Cifar':
            data, model =  CIFAR(), CIFARModel("models/cifar", sess)


        def _model_fn(x, logits=False):
            ybar, logits_ = model.predict(x)
            if logits:
                return ybar, logits_
            return ybar

        
        if FLAGS.dataset == 'MNIST':
            x_adv = fgsm(_model_fn, x, epochs=9, eps=0.02)
        elif FLAGS.datset == 'Cifar':
            x_adv = fgsm(_model_fn, x, epochs=4, eps=0.01)

        X_adv_test = attack(x_adv, data.test_data, data.test_labels, sess)
        X_adv_train = attack(x_adv, data.train_data, data.train_labels, sess)

        np.save('adversarial_outputs/fgsm_train_' + FLAGS.dataset.lower() + '.npy', X_adv_train)
        np.save('adversarial_outputs/fgsm_test_' + FLAGS.dataset.lower() + '.npy', X_adv_test)
        print("Legit/Adversarial training set")
        model.evaluate(data.train_data, data.train_labels)
        model.evaluate(X_adv_train, data.train_labels)
        
        print("Legit/Adversarial test set")
        model.evaluate(data.test_data, data.test_labels)
        model.evaluate(X_adv_test, data.test_labels)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
