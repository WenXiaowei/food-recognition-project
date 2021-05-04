from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, MaxPool2D, \
    Conv2DTranspose, concatenate, Activation

import tensorflow as tf


def layer(inputs, n_filters, filter_size=3, dropout_p=0.2):
    """
    A normal layer of the Tiramisu network
    :param inputs: the previous layers of the network
    :param n_filters: number of filters used in the 2D Conv operation
    :param filter_size: the size of the filter used in the 2D Conv
    :param dropout_p: the DropOut of the DropOut layer
    :return: the block of layers [BatchNormalization, ReLu, Conv2D, Dropout]
    """
    # tf.shape(inputs)

    l = BatchNormalization()(inputs)
    l = Conv2D(n_filters, filter_size, padding='same', kernel_initializer="he_uniform")(l)
    if dropout_p != 0.0:
        return Dropout(dropout_p)(l)
    return l


def transition_down(inputs, n_filters, dropout_p=0.2):
    """
    A transition down block of Tiramisu Network
    :param inputs: previous layers of the network
    :param dropout_p: the drop-out rate
    :param n_filters: number of filters used in the 2D Conv operation
    :return: the block of layers [BatchNormalization, ReLu, Conv2D, Dropout, MaxPool2D]
    """
    l = layer(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = MaxPool2D(2)(l)

    return l


def transition_up(skip_connection, block_to_up_sample, n_filters_keep):
    """
    A transition up block of Tiramisu Network
    :param skip_connection:
    :param n_filters_keep:
    :param block_to_up_sample:
    :return: the block of layers [Conv2DTranspose, block_to_up_sample] concatenated together
    """
    l = concatenate(block_to_up_sample, axis=-1)
    l = Conv2DTranspose(n_filters_keep, kernel_size=3, strides=2, kernel_initializer="he_uniform")(l)
    print(f"block_to_up_sample shape {tf.shape(block_to_up_sample)}")
    print(f"l shape {tf.shape(l)}")
    print(f"skip_connection shape {tf.shape(skip_connection)}")
    return concatenate([l, skip_connection], axis=-1)  # cropping=[None, None, 'center', 'center']


def soft_max(inputs, n_classes):
    """
    adding the final activation layer to the network
    :param inputs:
    :param n_classes:
    :return: the block of layer [Conv2D (1x1), Activation(softmax)]
    """
    net_layer = Conv2D(n_classes, kernel_size=1, padding="same", kernel_initializer="he_uniform")(inputs)
    return Activation("softmax")(net_layer)
