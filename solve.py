import os
import sys
import numpy as np
import scipy.io
import scipy.misc
import cv2
import tensorflow as tf
import h5py as h5
import pandas as pd 

f = h5py.File('trained_model-11-19.h5', 'r')

file_path = 'cac.csv'
# Constants for the image input and output.
# Output folder for the images.
OUTPUT_DIR = 'output/'
# Style image 
STYLE_IMAGE = 'images/StarryNight.jpg'
# Content image to use.
CONTENT_IMAGE = 'images/hongkong.jpg'
# Image dimensions constants. 
#ls = cv2.imread(CONTENT_IMAGE)
#print(ls.shape)
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
COLOR_CHANNELS = 3
# Noise ratio. Percentage of weight of the noise for intermixing with the
# content image.
NOISE_RATIO = 0.6
# Number of iterations to run.
ITERATIONS = 20
# Constant to put more emphasis on content loss.
BETA = 5
# Constant to put more emphasis on style loss.
ALPHA = 100
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
# The mean to subtract from the input to the VGG model. This is the mean that
# when the VGG was used to train. Minor changes to this will make a lot of
# difference to the performance of model.
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

def generate_noise_image(content_image, noise_ratio = NOISE_RATIO):
    """
    Returns a noise image intermixed with the content image at a certain ratio.
    """
    noise_image = np.random.uniform(
            -20, 20,
            (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')
    # White noise image from the content representation. Take a weighted average
    # of the values
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image
''' def crop_center(img,cropx,cropy):
    y,x,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]
 '''
K.set_image_data_format('channels_last')




# def CapsNet(input_shape, n_class, routings):
#     """
#     A Capsule Network on MNIST.
#     :param input_shape: data shape, 3d, [width, height, channels]
#     :param n_class: number of classes
#     :param routings: number of routing iterations
#     :return: Two Keras Models, the first one used for training, and the second one for evaluation.
#             `eval_model` can also be used for training.
#     """
#     x = layers.Input(shape=input_shape)

#     # Layer 1: Just a conventional Conv2D layer
#     conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

#     # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
#     primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

#     # Layer 3: Capsule layer. Routing algorithm works here.
#     digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
#                              name='digitcaps')(primarycaps)

#     # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
#     # If using tensorflow, this will not be necessary. :)
#     out_caps = Length(name='capsnet')(digitcaps)

#     # Decoder network.
#     y = layers.Input(shape=(n_class,))
#     masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
#     masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

#     # Shared Decoder model in training and prediction
#     decoder = models.Sequential(name='decoder')
#     decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
#     decoder.add(layers.Dense(1024, activation='relu'))
#     decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
#     decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    



def load_image(path):
    image = scipy.misc.imread(path)
    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    #image = crop_center(image, 400, 400)
    image = np.reshape(image, ((1,) + image.shape))
    # Input to the VGG model expects the mean to be subtracted.
    image = image - MEAN_VALUES
    return image

def multiplier(image, file_path):
    # import pandas as pd
    df = pd.read_csv(read_csv)
    mat = df.iloc[1:, 1:]
    mat_val = mat.values
    image = np.einsum('ij,jkl->ikl',mat_val,image)
    

def save_image(path, image):
    # Output should add back the mean.
    image = image + MEAN_VALUES
    # Get rid of the first useless dimension, what remains is the image.
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

def load_vgg_model(path):
    

    vgg = scipy.io.loadmat(path)

    vgg_layers = vgg['layers']
    def _weights(layer, expected_layer_name):
        """
        Return the weights and bias from the VGG model for a given layer.
        """
        W = vgg_layers[0][layer][0][0][0][0][0]
        b = vgg_layers[0][layer][0][0][0][0][1]
        layer_name = vgg_layers[0][layer][0][0][-2]
        assert layer_name == expected_layer_name
        return W, b

    def _relu(conv2d_layer):
        """
        Return the RELU function wrapped over a TensorFlow layer. Expects a
        Conv2d layer input.
        """
        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):
        """
        Return the Conv2D layer using the weights, biases from the VGG
        model at 'layer'.
        """
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(
            prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer, layer_name):
        """
        Return the Conv2D + RELU layer using the weights, biases from the VGG
        model at 'layer'.
        """
        return _relu(_conv2d(prev_layer, layer, layer_name))

    def _avgpool(prev_layer):
        """
        Return the AveragePooling layer.
        """
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def caps_call(caps_layers,shape=[800,600,3]):

    # Constructs the graph model.
    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)), dtype = 'float32')
    graph['caps'] =     _caps_relu(graph['input'],0 , 'caps' )
    graph['conv1_1']  = _conv2d_relu(graph['caps'], 1, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    return graph

def content_loss_func(sess, model):
    """
    Content loss function as defined in the paper.
    """
    def _content_loss(p, x):
        # N is the number of filters (at layer l).
        N = p.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = p.shape[1] * p.shape[2]

        return (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))
    return _content_loss(sess.run(model['conv4_2']), model['conv4_2'])

def style_loss_func(sess, model):
    """
    Style loss function as defined in the paper.
    """
    def _gram_matrix(F, N, M):
        """
        The gram matrix G.
        """
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    def _style_loss(a, x):
        """
        The style loss calculation.
        """
        # N is the number of filters (at layer l).
        N = a.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = a.shape[1] * a.shape[2]
        # A is the style representation of the original image (at layer l).
        A = _gram_matrix(a, N, M)
        # G is the style representation of the generated image (at layer l).
        G = _gram_matrix(x, N, M)
        result = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
        return result

    # Layers to use. We will use these layers as advised in the paper.
    # To have softer features, increase the weight of the higher layers
    # (conv5_1) and decrease the weight of the lower layers (conv1_1).
    # To have harder features, decrease the weight of the higher layers
    # (conv5_1) and increase the weight of the lower layers (conv1_1).
    layers = [
        ('conv1_1', 0.5),
        ('conv2_1', 1.0),
        ('conv3_1', 1.5),
        ('conv4_1', 3.0),
        ('conv5_1', 4.0),
    ]

    E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in layers]
    W = [w for _, w in layers]
    loss = sum([W[l] * E[l] for l in range(len(layers))])
    return loss


if __name__ == '__main__':
    with tf.Session() as sess:
        # Load the images.
        content_image = load_image(CONTENT_IMAGE)
        style_image = load_image(STYLE_IMAGE)
        # Load the model.
        model = load_vgg_model(VGG_MODEL)

        # Generate the white noise and content presentation mixed image
        # which will be the basis for the algorithm to "paint".
        input_image = generate_noise_image(content_image)

        sess.run(tf.initialize_all_variables())
        # Construct content_loss using content_image.
        sess.run(model['input'].assign(content_image))
        content_loss = content_loss_func(sess, model)

        # Construct style_loss using style_image.
        sess.run(model['input'].assign(style_image))
        style_loss = style_loss_func(sess, model)

        # Instantiate equation 7 of the paper.
        total_loss = BETA * content_loss + ALPHA * style_loss
        # The content is built from one layer, while the style is from five
        # layers. Then we minimize the total_loss, which is the equation 7.
        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(total_loss)

        sess.run(tf.initialize_all_variables())
        sess.run(model['input'].assign(input_image))
        for it in range(ITERATIONS):
            sess.run(train_step)

            mixed_image = sess.run(model['input'])
            print('Iteration %d' % (it))
            print('sum : ', sess.run(tf.reduce_sum(mixed_image)))
            print('cost: ', sess.run(total_loss))

            if not os.path.exists(OUTPUT_DIR):
                os.mkdir(OUTPUT_DIR)

            filename = 'output/%d.png' % (it)
            save_image(filename, mixed_image)