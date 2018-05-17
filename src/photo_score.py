from src import utils
import mxnet as mx
import base64
from mxnet import image
from mxnet import nd
from collections import namedtuple

train_augs = [
    image.HorizontalFlipAug(1.),
    image.RandomCropAug((224, 224))
]

test_augs = [
    image.CenterCropAug((224, 224))
]

def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2, 0, 1))
    return data, nd.array([label]).asscalar().astype('float32')

net = mx.mod.Module.load('../model/food', 0, True)
net.bind(data_shapes=[('data', (64, 3, 244, 244))])
batch = namedtuple('batch', ['data'])

def score_image(image_base64_string):
    with open('target_image.jpg', 'wb') as f:
        f.write(base64.b64decode(image_base64_string))
        f.close()
    with open('target_image.jpg', 'rb') as f:
        img = image.imdecode(f.read())
    data, _ = transform(img, -1, test_augs)
    data.transpose((1, 2, 0)).asnumpy()/255
    data = data.expand_dims(axis=0)
    net.forward(batch([data]), is_train=False)
    out = net.get_outputs()[0]
    out = nd.SoftmaxActivation(out)
    return int(out[0][1].asscalar() * 100)
