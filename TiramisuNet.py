import tensorflow
from tensorflow.python.keras.models import Model

from tiramisu_layers import *
from tensorflow.keras.layers import concatenate, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
from data_generators import *


def TiramisuNet(
        input_shape=(None, None, 3),
        n_classes=274,
        n_filters_first_conv=48,
        n_layers_per_block=None,
        growth_rate=16,
        n_pool=5,
        dropout_p=0.2
):
    if n_layers_per_block is None:
        n_layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]

    inputs = Input(input_shape)

    print(f"{tf.shape(inputs)}")
    # We perform a first convolution. All the features maps will be stored in the tensor called stack (the Tiramisu)
    stack = Conv2D(n_filters_first_conv, kernel_size=3, padding='same', kernel_initializer="he_uniform")(inputs)
    # The number of feature maps in the stack is stored in the variable n_filters
    n_filters = n_filters_first_conv

    #####################
    # Downsampling path #
    #####################

    skip_connection_list = []

    for i in range(n_pool):
        # Dense Block
        for j in range(n_layers_per_block[i]):
            # Compute new feature maps
            l = layer(stack, growth_rate, dropout_p=dropout_p)
            # And stack it : the Tiramisu is growing
            stack = concatenate([stack, l], axis=-1)  # problem?
            n_filters += growth_rate
        # At the end of the dense block, the current stack is stored in the skip_connections list
        skip_connection_list.append(stack)
        # Transition Down
        stack = transition_down(stack, n_filters, dropout_p)

    skip_connection_list = skip_connection_list[::-1]

    #####################
    #     Bottleneck    #
    #####################

    # We store now the output of the next dense block in a list. We will only upsample these new feature maps
    block_to_upsample = []

    # Dense Block
    for j in range(n_layers_per_block[n_pool]):
        l = layer(stack, growth_rate, dropout_p=dropout_p)
        block_to_upsample.append(l)
        stack = concatenate([stack, l], axis=-1)

    #######################
    #   Up-sampling path  #
    #######################
    # print(f"Before Upsampling {stack.type_spec.shape}")
    for i in range(n_pool):
        # print(f"block {i + 1}")
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = transition_up(skip_connection_list[i], block_to_upsample, n_filters_keep)

        # Dense Block
        block_to_upsample = []
        for j in range(n_layers_per_block[n_pool + i + 1]):
            # print(f"layer {j + 1} ")
            l = layer(stack, growth_rate, dropout_p=dropout_p)
            block_to_upsample.append(l)
            stack = concatenate([stack, l], axis=-1)

    #####################
    #      Softmax      #
    #####################

    return Model(inputs=inputs, outputs=soft_max(stack, n_classes))


model = TiramisuNet(
    # input_shape=(480,480, 3),
    input_shape=(460, 460, 3)
    , n_classes=274
)
model.summary()

model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#
history = model.fit(get_train_data_generator(),
                    steps_per_epoch=100,
                    epochs=10,
                    validation_data=get_val_data_generator(),
                    validation_steps=50)

# to compile with RMSprop
# learning rate 1e-3
# exponential decay of 0.995 after each epoch
# regularize with a weight decay of 1e-4
# dropout rate 0.2
# grow_rate = 16
# n_pool = 5
