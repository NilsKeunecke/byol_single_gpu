import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Dense, Add, Concatenate
from tensorflow.keras.layers import ZeroPadding2D, Input, MaxPooling2D, AveragePooling2D, Flatten
from tensorflow.keras import activations
from tensorflow.keras import Model
from tensorflow_addons.layers import GroupNormalization


def ws_reg(kernel):
    kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
    kernel = kernel - kernel_mean
    kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
    kernel = kernel / (kernel_std + 1e-5)
    # return kernel


def res_identity(x, filters):
    # resnet block where dimension does not change.
    # The skip connection is just simple identity connection
    # we will have 3 blocks and then input will be added

    x_skip = x  # this will be used for addition with the residual block
    f1, f2 = filters

    # first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=ws_reg)(x)
    x = GroupNormalization(groups=16, axis=-1)(x)
    # x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # second block # bottleneck (but size kept same with padding)
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=ws_reg)(x)
    x = GroupNormalization(groups=16, axis=-1)(x)
    # x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # third block activation used after adding the input
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=ws_reg)(x)
    #x = GroupNormalization(groups=16, axis=-1)(x)
    # x = BatchNormalization()(x)

    # add the input
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x


def res_conv(x, s, filters):
    ''' here the input size changes'''
    x_skip = x
    f1, f2 = filters

    # first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=ws_reg)(x)
    # when s = 2 then it is like downsizing the feature map
    x = GroupNormalization(groups=16, axis=-1)(x)
    # x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # second block
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=ws_reg)(x)
    x = GroupNormalization(groups=16, axis=-1)(x)
    # x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # third block
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=ws_reg)(x)
    #x = GroupNormalization(groups=16, axis=-1)(x)
    # x = BatchNormalization()(x)

    # shortcut
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=ws_reg)(x_skip)
    #x_skip = GroupNormalization(groups=16, axis=-1)(x_skip)
    # x_skip = BatchNormalization()(x_skip)

    # add
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x


def resnet50(train_im):
    input_im = Input(shape=(train_im[0], train_im[1], train_im[2]))  # cifar 10 images size
    x = ZeroPadding2D(padding=(3, 3))(input_im)

    # 1st stage
    # here we perform maxpooling, see the figure above

    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), kernel_regularizer=ws_reg)(x)
    #x = GroupNormalization(groups=16, axis=-1)(x)
    # x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 2nd stage
    # from here on only conv block and identity block, no pooling

    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))

    # 3rd stage

    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))

    # 4th stage

    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))

    # 5th stage

    x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))

    # ends with average pooling and dense connection

    x = AveragePooling2D((2, 2), padding='same')(x)

    x = Flatten()(x)
    # x = Dense(10, activation='softmax', kernel_initializer='he_normal')(x) #multi-class

    # define the model

    model = Model(inputs=input_im, outputs=x, name='Resnet50')

    return model
