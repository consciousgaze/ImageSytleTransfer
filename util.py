import caffe, pickle, subprocess, os
import tensorflow as tf
from tensorflow.python.ops import nn
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format


def get_pooling_types_dict():
    """
        Get dictionary mapping pooling type number to type name
    """
    desc = caffe_pb2.PoolingParameter.PoolMethod.DESCRIPTOR
    d = {}
    for k, v in desc.values_by_name.items():
        d[v.number] = k
    return d

def getVGGCaffeWeights():
    get_model = 'curl http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel -o models/weight.caffemodel'
    p = subprocess.Popen(get_model.split(), stdout = subprocess.PIPE)
    p.wait()

def convertRELU(inLayer, clayer, nlayer):
    #return tf.nn.relu(inLayer, name = clayer.name)
    rlt = {}
    rlt['type'] = 'relu'
    rlt['inlayer'] = inLayer['name']
    rlt['name'] = clayer.name
    return rlt

def convertConv(inLayer, clayer, nlayer):
    # get padding
    if clayer.convolution_param.pad_h != 0 and clayer.convolution_param.pad == 0:
        padding = 'VALID'
    else:
        padding = 'SAME'
    # get strides
    if clayer.convolution_param.stride_h != 0:
        stride = [1, clayer.convolution_param.stride_h, clayer.convolution_param.stride_w, 1]
    elif len(clayer.convolution_param.stride) > 0:
        stride = [1, clayer.convolution_param.stride[0], clayer.convolution_param.stride[0], 1]
    else:
        stride = [1, 1, 1, 1]

    '''
        caffe's order is [batch, channel, hight, width]
        tensor's order is [batch, hight, width, channel]
    '''
    filters = nlayer.blobs[0].data[:].transpose((2, 3, 1, 0)).astype(np.float64)
    output = {}
    output['type'] = 'conv'
    output['inlayer'] = inLayer['name']
    output['filters'] = filters
    output['stride'] = stride
    output['padding'] = padding
    output['name'] = clayer.name
    if len(nlayer.blobs) == 2:
        biases = nlayer.blobs[1].data[:].astype(np.float64)
        #outputs = tf.nn.conv2d(inLayer, filters, stride, padding=padding)
        #outputs = tf.nn.bias_add(outputs, biases, name = clayer.name)
        output['bias'] = biases
    #else:
    #    outputs = tf.nn.conv2d(inLayer, filters, stride, padding=padding, name = clayer.name)
    return output


def convertPool(inLayer, clayer, nlayer):
    # get pooling type
    poolType = get_pooling_types_dict()[clayer.pooling_param.pool]
    # get pooling type
    if clayer.pooling_param.pad != 0:
        padding = 'VALID'
    else:
        padding = 'SAME'
    # get kernal size
    if clayer.pooling_param.kernel_h != 0:
        kernel = [1, clayer.pooling_param.kernel_h, clayer.pooling_param.kernel_w, 1]
    else:
        kernel = [1, clayer.pooling_param.kernel_size, clayer.pooling_param.kernel_size, 1]
    # get stride
    if clayer.pooling_param.stride_h != 0:
        stride = [1, clayer.pooling_param.stride_h, clayer.pooling_param.stride_w, 1]
    else:
        stride = [1, clayer.pooling_param.stride, clayer.pooling_param.stride, 1]

    rlt = {}
    rlt['type'] = 'pool'
    rlt['inlayer'] = inLayer['name']
    rlt['ksize'] = kernel
    rlt['strides'] = stride
    rlt['padding'] = padding
    rlt['name'] = clayer.name
    # construct pooling layer
    if poolType == "AVE":
        rlt['pool_type'] = 'AVE'
        #return tf.nn.avg_pool(inLayer, ksize = kernel, strides = stride,
        #                                    padding = padding, name = clayer.name)
    elif poolType == 'MAX':
        rlt['pool_type'] = 'MAX'
        #return tf.nn.max_pool(inLayer, ksize = kernel, strides = stride,
        #                                    padding = padding, name = clayer.name)
    elif poolType == 'STOCHASTIC':
        raise NotImplementedError("Stochastic pooling layer is not implemented yet")
    else:
        raise ValueError(poolType + " is not valid for pooling layer type.")
    return rlt



def convertVGGModel():
    converter = {
                    "ReLU" : convertRELU,
                    "Convolution" : convertConv,
                    "Pooling" : convertPool
                }

    MODEL_FILE = "models/model.prototxt"
    MODEL_WEIGHT = "models/weight.caffemodel"
    MEAN_FILE = "models/mean.npy"

    if not os.path.isfile(MODEL_WEIGHT):
        getVGGCaffeWeights()

    cnet = caffe_pb2.NetParameter()
    text_format.Merge(open(MODEL_FILE).read(), cnet)
    nnet = caffe.Net(MODEL_FILE, MODEL_WEIGHT, caffe.TEST)

    shape = (1,224,224, 3)
    outputs = {}
    pre = {'name':'data'}
    for idx, layer in enumerate(cnet.layer):
        assert nnet._layer_names[idx + 1] == layer.name
        cur = converter[layer.type](pre, layer, nnet.layers[idx + 1])
        outputs[layer.name] = cur
        pre = cur

    pickle.dump(outputs, open('models/tfmodel.p', 'wb'))

