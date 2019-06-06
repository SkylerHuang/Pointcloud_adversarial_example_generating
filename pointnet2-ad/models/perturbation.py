import tensorflow as tf
import numpy as nu

def perturbation_point_xyz(point_cloud):
    B, N, C = point_cloud.shape
    epsilon = tf.get_variable(name='epsilon', shape = [B,N,C], initializer = tf.contrib.layers.xavier_initializer(),dtype= tf.float32 , trainable = False)

    pert_pc = tf.add(point_cloud, epsilon)
    return pert_pc, epsilon
