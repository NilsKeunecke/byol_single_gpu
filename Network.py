import tensorflow as tf
import tensorflow_addons as tfa
import ResNet_GN_WS


def build_model(img_dims, use_GN_WS=True, online=True):
    """The online model using either GN or BN depending on setup"""

    # The Encoder
    if not use_GN_WS:
        resnet_model = tf.keras.applications.ResNet50(
            include_top=False, weights=None, input_tensor=None,  # pytorch benutzt pretrained resnet???
            input_shape=(32, 32, 3), pooling='avg')
    else:
        resnet_model = ResNet_GN_WS.resnet50(img_dims)

    # The Projector
    x = tf.keras.layers.Dense(4096)(resnet_model.output)

    if not use_GN_WS:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tfa.layers.GroupNormalization(groups=16, axis=-1)(x)

    x = tf.keras.layers.Activation('relu')(x)
    if online:
        x = tf.keras.layers.Dense(256)(x)
    else:
        output = tf.keras.layers.Dense(256)(x)

    if online:
        # The Predictor
        x = tf.keras.layers.Dense(4096)(x)
        if not use_GN_WS:
            x = tf.keras.layers.BatchNormalization()(x)
        else:
            x = tfa.layers.GroupNormalization(groups=16, axis=-1)(x)
        x = tf.keras.layers.Activation('relu')(x)
        output = tf.keras.layers.Dense(256)(x)

    return tf.keras.Model(inputs=resnet_model.inputs, outputs=output)