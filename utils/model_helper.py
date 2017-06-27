import numpy as np 
import urllib
import os

def getModel(prefix, code, model_dir):
    download(prefix + '-symbol.json', model_dir)
    download(prefix + '-%04d.params' % code, model_dir)

# obtain the pre-trained model
def download(url, model_dir):
    filename = url.split('/')[-1]
    if not os.path.exists(model_dir + filename):
        urllib.urlretrieve(url, model_dir + filename)

def specContext(param, ctx):
    """
    This func specifies the device context(computation source:CPU/GPU)
    of the NDArray
    
    Inputs:
        - param: dict of str to NDArray
        - ctx: the device context(Context or list of Context)
    
    Returns:
        None
    """
    for k, v in param.items():
        param[k] = v.as_in_context(ctx)
        
    return
    
def loadPretrainedModel(prefix, epoch, ctx = None):
    """
    This func is a wrapper of the mx.model.load_checkpoint. It can 
    also specify the context(computation source:CPU/GPU) that will
    the params
    
    Inputs:
        - prefix: string indicating prefix of model name
        - epoch: int indicating epoch number
        - ctx: the device context(Context or list of Context)
    
    Returns:
        - arg_params: dict of str to NDArray of net's weights
        - aux_params: dict of str to NDArray of net's auxiliary states
    """
    
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    logging.info('The pretrained model has been loaded successfully!')
    if ctx:
        specContext(arg_params, ctx)
        specContext(aux_params, ctx)
    return sym, arg_params, aux_params

def refactorModel(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    This func is to replace the last fully-connected layer of the pre-trained model
    with our defined last fully-connected layer
    
    Inputs:
        - symbol: the pretrained network symbol
        - arg_params: the argument parameters of the pretrained model
        - num_classes: the number of classes for the fine-tune datasets
        - layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)