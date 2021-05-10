from tensorflow import keras
import tensorflow as tf
import keras.backend as K
import numpy as np

scale = 6


def set_scale(new_scale):
    global scale
    scale = new_scale


def doomloss(y_true, y_pred):
    doom_true = 1 - K.tanh(y_true * 0.80642766 / scale)
    doom_pred = 1 - K.tanh(y_pred * 0.80642766 / scale)
    weight = (doom_true + doom_pred) / 2
    return K.mean(weight * K.square(y_true - y_pred))


def NMSE(y_true, y_pred):
    return keras.losses.MSE(y_true, y_pred) / keras.losses.MSE(y_true, K.cast(scale, "float32"))


def pairwise_contrastive_loss(y_true, y_pred):
    mask = K.less(y_true, scale)
    return K.mean(K.switch(mask, K.square(y_pred), K.square(K.maximum(scale - y_pred, 0))))


def binary_cross_entropy(y_true, y_pred):
    mask = K.less(y_true, scale)
    return keras.losses.binary_crossentropy(mask, y_pred)
