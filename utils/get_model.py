import numpy as np 
import urllib

def getModel(prefix, code, model_dir):
    download(prefix + '-symbol.json', model_dir)
    download(prefix + '-%04d.params' % code, model_dir)

# obtain the pre-trained model
def download(url, model_dir):
    filename = url.split('/')[-1]
    if not os.path.exists(model_dir + filename):
        urllib.urlretrieve(url, model_dir + filename)