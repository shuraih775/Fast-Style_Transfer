import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, concatenate, MaxPooling2D, Input

vgg = keras.applications.VGG19(include_top = False,weights = 'imagenet')
vgg.trainable = False

def get_features(images, model, layers = None):
#     layers : should be dict type argument
    if layers is None:
        layers = {'1':'block1_conv1','4':'block2_conv1','7':'block3_conv1','12':'block4_conv1',
                  '17':'block5_conv1'}
    else:
        layers = layers
    features = []
    for layer in model.layers:
        images = layer(images)
        if layer.name in layers.values():
            features.append(images)

    return features



class MirrorPaddingLayer(keras.layers.Layer):
    def __init__(self, pad_size=1, **kwargs):
        super(MirrorPaddingLayer, self).__init__(**kwargs)
        self.pad_size = pad_size
        self.trainable = False

    def build(self, input_shape):
        super(MirrorPaddingLayer, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3]
        padded_height = height + 2 * self.pad_size
        padded_width = width + 2 * self.pad_size

        paddings = tf.constant([[0, 0], [self.pad_size, self.pad_size], [self.pad_size, self.pad_size], [0, 0]])
        padded_batch = tf.pad(inputs, paddings, mode='REFLECT')

        return padded_batch

    def compute_output_shape(self, input_shape):
        batch_size, height, width, channels = input_shape
        padded_height = height + 2 * self.pad_size
        padded_width = width + 2 * self.pad_size
        return (batch_size, padded_height, padded_width, channels)



def Block(kernel_size, filters):
    def block(x_in):
        x_in = MirrorPaddingLayer()(x_in)
        x = Conv2D(kernel_size=kernel_size, filters=filters, padding='same', kernel_initializer='glorot_uniform')(x_in)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x
    return block



def join(small_k_input, large_k_input):
    sk_input =tf.image.resize(small_k_input, size=(large_k_input.shape[1],large_k_input.shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    sk_input = BatchNormalization()(sk_input)
    lk_input = BatchNormalization()(large_k_input)
    concatenated = concatenate([sk_input, lk_input], axis=-1)
    return concatenated



def Conv_block(filters):
    def conv_block(x_in):
        x = x_in
        x = Block((3, 3), filters)(x)
        x = Block((3, 3), filters)(x)
        x = Block((1, 1), filters)(x)
        return x
    return conv_block



def fast_style_generator(batch_size=16):
    content_images = Input(shape=(224, 224, 3), batch_size=batch_size)
    content_features = get_features(content_images, vgg, layers={'18':'block5_conv2'})  

    zinput_5 = tf.random.normal(shape=(batch_size, 14, 14, 8))
    zinput_4 = tf.random.normal(shape=(batch_size, 28, 28, 8))
    zinput_3 = tf.random.normal(shape=(batch_size, 56, 56, 8))
    zinput_2 = tf.random.normal(shape=(batch_size, 112, 112, 8))
    zinput_1 = tf.random.normal(shape=(batch_size, 224, 224, 8))

    zinput_5 = concatenate([zinput_5, content_features[0]], axis=-1)
    z5 = Conv_block(8)(zinput_5)
    z4 = Conv_block(8)(zinput_4)
    z4 = join(z5, z4)

    z4 = Conv_block(16)(z4)
    z3 = Conv_block(8)(zinput_3)
    z3 = join(z4, z3)

    z3 = Conv_block(24)(z3)
    z2 = Conv_block(8)(zinput_2)
    z2 = join(z3, z2)

    z2 = Conv_block(32)(z2)
    z1 = Conv_block(8)(zinput_1)
    z1 = join(z2, z1)

    z1 = Conv_block(40)(z1)
    z1 = Conv_block(40)(z1)
    z1 = Block((1, 1), 40)(z1)
    outputs = Block((1, 1), 3)(z1)

    return keras.models.Model(content_images, [outputs, z1, z2, z3, z4, z5], name='fast_style_generator')



def gram_matrix(feature_maps):
    batch_size, height, width, channels = feature_maps.shape
    reshaped_feature_maps = np.reshape(feature_maps, (batch_size, height * width, channels))
    gram = np.matmul(reshaped_feature_maps, reshaped_feature_maps.transpose(0, 2, 1))
    gram /= height * width * channels
    return gram



def model_loss(x_style, x_contents, model_output):
    target_style_features = get_features(x_style, vgg)
    target_content_features = get_features(x_contents, vgg, layers={'13':'block4_conv2'})

    gram_targets = [gram_matrix(feature_map) for feature_map in target_style_features]

    loss = 0

    # Texture loss
    for i in range(len(gram_targets)):
        loss += np.sum(np.square(gram_targets[i] - model_output[i+1][0]))

    # Content loss
    for i in range(len(target_content_features)):
        loss += np.sum(np.square(target_content_features[i] - model_output[4][i]))

    return loss





def train(content_images, style_image, model, n_steps=2000, batch_size=16):

    style_images = [style_image] * len(content_images)

    print(content_images)
    print(tf.shape(style_images))

    dataset = tf.data.Dataset.from_tensor_slices((content_images, style_images))
    dataset = dataset.shuffle(len(content_images)).batch(batch_size)
    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=200,
        decay_rate=0.7,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


    def compute_loss(content_batch, style_batch):
        with tf.GradientTape() as tape:
            model_outputs = model(content_batch)
            loss = model_loss(style_batch, content_batch, model_outputs)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss


    for step, (content_batch, style_batch) in enumerate(dataset.take(n_steps), 1):
        loss = compute_loss(content_batch, style_batch)
        if step % 100 == 0:
            print(f"Step {step}/{n_steps}, Loss: {loss.numpy()}")



