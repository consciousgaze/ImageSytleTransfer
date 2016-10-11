import pickle, os, argparse
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from util import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from skimage.transform import resize
from scipy.misc import imread

INCEP_CONVLAYERS = {'conv2d0' : .2,
                    'conv2d1' : .2,
                    'conv2d2' : .2,
                    'mixed3a' : .1,
                    'maxpool4' : .1,
                    'maxpool10' : .1,
                    'avgpool0' : .1}

VGG_CONVLAYERS = { 'conv5_1' : 0.1,
                   'conv4_1' : 0.1,
                   'conv3_1' : 0.2,
                   'conv2_1' : 0.3,
                   'conv1_1' : 0.3 }

def parseArg():
    parser = argparse.ArgumentParser(description='Take style and content image')
    parser.add_argument('--style', help='The path to style image.', default = "starry_night.jpg")
    parser.add_argument('--content', help='The path to content image.', default = "full_moon.jpg")
    parser.add_argument('--max_iter', help='The maximum iterations the optimization funciton you want to run.',
                        default=5000)
    parser.add_argument('--out_file', help='The output file name. ouput.jpg is the default.',
                        default='output.jpg')
    return parser.parse_args()

class TensorflowImageStyleConverter:
    def __init__(self,
                 graphData = 'models/tensorflow_inception_graph.pb',
                 vgg_model = 'models/tfmodel.p'):
        self.vgg_model = vgg_model
        self.styleWeight = 7e4
        self.contentWeight = 1
        self.graphData = graphData

    def loadVGGModel(self, init, scope):
        self.convLayers = VGG_CONVLAYERS
        self.secConvLayer = 'conv4_2'

        def loadConvLayer(inLayer, params):
            filters = params['filters']
            stride = params['stride']
            padding = params['padding']
            name = '{}/{}'.format(scope, params['name'])
            if 'bias' in params:
                biases = params['bias']
                tmp = tf.nn.conv2d(inLayer, filters, stride, padding=padding)
                return tf.nn.bias_add(tmp, biases, name = name)
            else:
                return tf.nn.conv2d(inLayer, filters, stride, padding=padding, name = name)

        def loadPoolLayer(inLayer, params):
            kernel = params['ksize']
            stride = params['strides']
            padding = params['padding']
            name = '{}/{}'.format(scope, params['name'])

            if params['pool_type'] == 'AVE':
                return tf.nn.avg_pool(inLayer, ksize = kernel, strides = stride,
                                      padding = padding, name = name)
            elif params['pool_type'] == 'MAX':
                return tf.nn.max_pool(inLayer, ksize = kernel, strides = stride,
                                      padding = padding, name = name)
            else:
                raise ValueError("Only average pooling and max pooling are accepted.")

        def loadRELULayer(inLayer, params):
             name = '{}/{}'.format(scope, params['name'])
             return tf.nn.relu(inLayer, name = 'name')

        def loadVGGLayer(name):
            if name == 'data':
                image = tf.Variable(init, trainable = True, name = '{}/inputImage'.format(scope))
                return image

            layer = params[name]
            inLayer = loadVGGLayer(layer['inlayer'])
            if layer['type'] == 'conv':
                rlt =  loadConvLayer(inLayer, layer)
            elif layer['type'] == 'pool':
                rlt =  loadPoolLayer(inLayer, layer)
            elif layer['type'] == 'relu':
                rlt = loadRELULayer(inLayer, layer)
            else:
                raise ValueError("Onye convolutional layer, pooling layer and relu layer are accepted.")
            return rlt

        params = pickle.load(open(self.vgg_model))
        g = tf.Graph()
        sess = tf.Session(graph = g)
        with sess.as_default():
            with g.as_default():
                loadVGGLayer('pool5')
                image = tf.all_variables()[0]
        return sess, image


    def loadInceptionModel(self, init, scope):
        self.convLayers = INCEP_CONVLAYERS
        self.secConvLayer = 'mixed4e'
        sess = tf.Session(graph = tf.Graph())
        with sess.as_default():
            with sess.graph.as_default():
                with gfile.GFile(self.graphData, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())

                image = tf.Variable(init, trainable = True, name = '{}/inputImage'.format(scope))
                tf.import_graph_def(graph_def, input_map={"input:0":image}, name = scope)
        return sess, image

    def getTensor(self, name, graph, scope):
        return graph.get_tensor_by_name(scope + '/' + name + ':0')

    def getLayers(self, layers,  sess, scope):
        layer_dic = {}
        with sess.as_default():
            with sess.graph.as_default():
                sess.run(tf.initialize_all_variables())
                for layer in layers:
                    op = self.getTensor(layer, sess.graph, scope)
                    rlt = op.eval()
                    layer_dic[layer] = self.getF(rlt)
        return layer_dic

    def getF(self, L):
        b, w, h, l = L.shape
        return L.reshape(w * h, l)

    def getG(self, F, sess):
        with sess.as_default():
            with sess.graph.as_default():
                return tf.matmul(F, F, transpose_a = True)

    def getLoss(self, styleLayers, contentLayers, sess):
        with sess.as_default():
            with sess.graph.as_default():
                loss = tf.constant(0.)
                for layer in styleLayers:
                    tmp = self.getTensor(layer, sess.graph, 'generated')
                    tmp = tf.reshape(tmp, styleLayers[layer].shape)
                    n, m = styleLayers[layer].shape
                    tmp = self.getG(tmp, sess)
                    tmp = tf.sub(tmp, self.getG(styleLayers[layer], sess))
                    tmp = tf.square(tmp)
                    tmp = tf.reduce_sum(tmp)
                    tmp = tf.mul(tmp, self.styleWeight)/n/n/m/m/4.
                    tmp = tf.mul(tmp, self.convLayers[layer])
                    loss = tf.add(loss, tmp)

                for layer in contentLayers:
                    tmp = self.getTensor(layer, sess.graph, 'generated')
                    tmp = tf.reshape(tmp, contentLayers[layer].shape)
                    tmp = tf.sub(tmp, contentLayers[layer])
                    tmp = tf.square(tmp)
                    tmp = tf.reduce_sum(tmp)
                    tmp = tf.mul(tmp, self.contentWeight)/2.
                    loss = tf.add(loss, tmp)
        return loss


    def initImage(self, initMode, content):
        return np.random.random_sample(content.shape).astype('float32') * 255 - 128

    def set_transformer(self, cont):
        h, w, c = cont.shape
        self.shape = (h, w)

    def preprocess(self, image):
        resized = resize(image, self.shape, preserve_range = True)
        resized = resized[np.newaxis]
        resized = resized.astype(np.float32)
        return resized - 117

    def deprocess(self, image):
        return np.clip(image + 117, 0, 255).astype(np.uint8)

    def transferImage(self, style, content, initMode = 'random', max_iter = 900):
        style = imread(style)
        content = imread(content)
        self.set_transformer(content)
        style = self.preprocess(style)
        content = self.preprocess(content)

        self.loadModel = self.loadVGGModel
        sess, _ = self.loadModel(content, 'content')
        contentLayers = self.getLayers([self.secConvLayer], sess, 'content')

        sess, _  = self.loadModel(style, 'style')
        styleLayers = self.getLayers(self.convLayers, sess, 'style')

        sess, image = self.loadModel(self.initImage(initMode, content), 'generated')
        with sess.as_default():
            with sess.graph.as_default():
                loss = self.getLoss(styleLayers, contentLayers, sess)
                gradients = tf.gradients(loss, image)
                step = tf.train.AdamOptimizer(3).minimize(loss, var_list = [image])
                tf.initialize_all_variables().run()
                previous = image.eval()
                for i in range(max_iter):
                    sess.run(step)
                    tmp = loss.eval()
                    if i%10 == 0:
                        print i, tmp,
                        tmp = image.eval()[0]
                        print np.square((tmp - previous)).sum()
                        previous = tmp
                generated = image.eval()[0]
        generated = self.deprocess(generated)
        return generated

def main():
    if not os.path.isfile('models/tfmodel.p'):
        print "VGG model parameters for Tensorflow is missing. Generating model parameters from Caffe model"
        convertVGGModel()
        print "VGG model parameters for Tensorflow model is generated."
        print "Since Caffe and Tensorflow cannot be run at same time, we will quit and please run your conversion again."
        return
    arg = parseArg()

    converter = TensorflowImageStyleConverter()
    generated = converter.transferImage(arg.style, arg.content, max_iter = arg.max_iter)
    plt.imshow(generated)
    plt.savefig(arg.out_file)

if __name__ == "__main__":
    main()
