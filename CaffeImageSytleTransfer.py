#/usr/bin/env python
import sys, caffe, argparse, pickle, math, os, logging, subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# the model is vgg19 in caffe model zoo
MODEL_FILE = "models/model.prototxt"
MODEL_WEIGHT = "models/weight.caffemodel"
MEAN_FILE = "models/mean.npy"
CONV_LAYER_NAMES = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
WEIGHTS = { 'conv5_1' : 0.1,
            'conv4_1' : 0.1,
            'conv3_1' : 0.2,
            'conv2_1' : 0.3,
            'conv1_1' : 0.3 }
get_model = 'curl http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel -o ' + MODEL_WEIGHT

def parseArg():
    parser = argparse.ArgumentParser(description='Take style and content image')
    parser.add_argument('--style', help='The path to style image.')
    parser.add_argument('--content', help='The path to content image.')
    parser.add_argument('--max_size', help='The maximum size of the output image in total contained pixles.',
                        default=512*512)
    parser.add_argument('--max_iter', help='The maximum iterations the optimization funciton you want to run.',
                        default=900)
    parser.add_argument('--out_file', help='The output file name. ouput.jpg is the default.',
                        default='output.jpg')
    return parser.parse_args()

class ImageStyleConverter:
    def __init__(self, styleImage, contentImage, max_iter = 900,
                 max_size = 512 * 512, out_file = 'output.jpg', use_gpu = True,
                 model_file = MODEL_FILE, model_weight = MODEL_WEIGHT, mean_file = MEAN_FILE):
        '''
            @styleImage: path to style image
            @contentImage: path to content iamge
            @max_iter: the maximum iterations for minimization
            @max_size: the maximum pixiles for input picture
                       if the picture is too large, it will be
                       rescaled this size
            @use_gpu: whether to use gpu
        '''
        if use_gpu:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.max_iter = max_iter
        self.out_file = out_file

        # load caffe net
        if not os.path.isfile(model_file):
            raise Exception("Model file does not found.")
        if not os.path.isfile(model_weight):
            raise Exception("Model weight does not found.")
        if not os.path.isfile(mean_file):
            raise Exception("Mean file does not found.")
        self.net = caffe.Net(model_file, model_weight, caffe.TEST)
        self.convLayerNames = CONV_LAYER_NAMES
        self.weights = WEIGHTS
        self.secConvLayerName = 'conv4_2'

        # struct model
        w, h, c = caffe.io.load_image(contentImage).shape
        act_size = h * w
        if act_size > max_size:
            r = max_size * 1. / act_size
            r = math.sqrt(r)
            h = int(r * h)
            w = int(r * w)
        self.data = self.net.blobs.keys()[0]
        self.net.blobs[self.data].reshape(1, c, w, h)
        self.c, self.w, self.h = self.net.blobs[self.data].data[0].shape

        # set transformer
        self.transformer = caffe.io.Transformer({self.data : self.net.blobs[self.data].data.shape})
        self.transformer.set_mean(self.data, np.load(MEAN_FILE).mean(1).mean(1))
        self.transformer.set_transpose(self.data, (2,0,1))
        self.transformer.set_channel_swap(self.data, (2,1,0))
        self.transformer.set_raw_scale(self.data, 255.0)

        # load images
        tmp = caffe.io.load_image(styleImage)
        self.styleImage = self.transformer.preprocess(self.data, tmp)
        tmp = caffe.io.load_image(contentImage)
        self.contentImage = self.transformer.preprocess(self.data, tmp)

        # style and content loss ratio
        self.styleLossWeight = 1e5
        self.contentLossWeight = 1

    def getConvlayers(self, image, layers):
        '''
            given an image, spit the dictionary of conv blobs
            with a key of the layer name, and value of the layer activations
            @ image: the image which is going to be computed.
        '''
        self.net.blobs[self.data].data[0] = image
        self.net.forward()
        convLayers = {}
        for layer in layers:
            convLayers[layer] = self.net.blobs[layer].data[0].copy()
        return convLayers

    def getG(self, convLayer):
        F = self.getF(convLayer)
        G = np.dot(F, F.T)
        return G

    def getF(self, F):
        if len(F.shape) < 3:
            return F.copy()
        tmp = F.copy()
        tmp = tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2])
        return tmp

    def getComputeGradient(self, style, content):
        def computeGradient(generated):
            generated = generated.reshape(self.c, self.w, self.h)
            layers = list(self.net.blobs.keys())
            layers.reverse()
            layerNum = len(layers)
            self.net.blobs[self.data].data[0] = generated
            self.net.forward()
            self.net.blobs[layers[0]].diff[:] = 0
            loss = 0
            for layer_idx in range(layerNum):
                layer = layers[layer_idx]
                g = self.net.blobs[layer].diff[0]
                F = self.net.blobs[layer].data[0]
                if layer == self.secConvLayerName:
                    g += (F - content[layer]) * (F > 0) * self.contentLossWeight
                    loss += np.square(F - content[layer]).sum() * 0.5 * self.contentLossWeight

                if layer in self.convLayerNames:
                    reshapedF = self.getF(F)
                    G = self.getG(F)
                    A = self.getG(style[layer])
                    n, h, w = F.shape
                    tmpg = np.dot(reshapedF.T/n/n/w/w/h/h, (G - A)).T * (reshapedF > 0)
                    tmpg = tmpg.reshape(g.shape)
                    g += tmpg * self.styleLossWeight * self.weights[layer]
                    loss += np.square(G - A).sum() * self.weights[layer] * self.styleLossWeight /n/n/h/h/w/w/4.
                if layer_idx + 2 < layerNum:
                    self.net.backward(start = layer, end = layers[layer_idx + 1])
                elif layer_idx == layerNum -1:
                    self.net.backward(start = layer, end = 'input')
                else:
                    self.net.backward(start = layer, end = None)
                    g = self.net.blobs[self.data].diff[0].copy()
                    break
            return loss, g.flatten().astype(np.float64)
        return computeGradient


    def generateInitImage(self, shape):
        c, h, w = shape
        tmp = np.random.rand(c, h, w)*100 # get random image of same size
        return self.transformer.preprocess(self.data, tmp)

    def styleConvert(self, generated, styleLayers, contentLayers,
                     alpha = 1, threshold = 1e-9):
        # prepare the funciotn to get loss and jacobbian
        computeGradient = self.getComputeGradient(styleLayers, contentLayers)
        # prepare optimization parameters
        mn = -self.transformer.mean[self.data][:,0,0]
        mx = mn + self.transformer.raw_scale["data"]
        bounds = [(mn[0], mx[0])]*(generated.size/3) + \
                 [(mn[1], mx[1])]*(generated.size/3) + \
                 [(mn[2], mx[2])]*(generated.size/3)
        minfn_args = {
            "method": 'L-BFGS-B', "jac": True, "bounds": bounds,
            "options": {"maxiter": self.max_iter, "disp": True}
        }

        # get results
        generated = minimize(computeGradient, generated, **minfn_args)
        generated = generated['x'].reshape(self.c, self.w, self.h)
        return generated

    def run(self):
        styleLayers = self.getConvlayers(self.styleImage, CONV_LAYER_NAMES)
        contentLayers = self.getConvlayers(self.contentImage, ['conv4_2'])
        generated = self.generateInitImage(self.contentImage.shape)
        generated = self.styleConvert(generated, styleLayers, contentLayers)
        self.show(generated)

    def show(self, image):
        image = self.transformer.deprocess(self.data, image)
        plt.imshow(image)
        plt.savefig(self.out_file)

if __name__ == '__main__':
    arg = parseArg()
    if not os.path.isfile(MODEL_FILE):
        raise Exception("""Model file does not found.
                           This example script is supposed to be run in the root directory.""")
    if not os.path.isfile(MODEL_WEIGHT):
        print "Trained weight file for vgg19 not found. Downloading..."
        p = subprocess.Popen(get_model.split(), stdout = subprocess.PIPE)
        p.wait()
    converter = ImageStyleConverter(arg.style, arg.content, arg.max_iter,
                                    arg.max_size, arg.out_file)
    converter.run()
