# Fine-tuning 通过微调来迁移学习

from mxnet import nd
from mxnet import image
from mxnet import gluon
from mxnet.gluon.model_zoo import vision as models
from mxnet import init
from src import utils
import matplotlib.pyplot as plt

data_dir = '../data'
model_dir = '../model'

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

train_imgs = gluon.data.vision.ImageFolderDataset(data_dir + '/food/train', transform=lambda X, y: transform(X, y, train_augs))
test_imgs = gluon.data.vision.ImageFolderDataset(data_dir + '/food/test', transform=lambda X, y: transform(X, y, test_augs))

pretrained_net = models.resnet18_v1(pretrained=True)

finetune_net = models.resnet18_v1(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())

def train(net, ctx, batch_size=64, epochs=1, learning_rate=0.01, wd=0.001):
    train_data = gluon.data.DataLoader(train_imgs, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(test_imgs, batch_size)
    # 确保net的初始化在ctx上
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    # 训练
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': wd})
    utils.train(train_data, test_data, net, loss, trainer, ctx, epochs)

ctx = utils.try_all_gpus()
train(finetune_net, ctx)

finetune_net.export(model_dir + '/food')

def classify_food(net, fname):
    with open(fname, 'rb') as f:
        img = image.imdecode(f.read())
    data, _ = transform(img, -1, test_augs)
    plt.imshow(data.transpose((1, 2, 0)).asnumpy()/255)
    data = data.expand_dims(axis=0)
    out = net(data.as_in_context(ctx[0]))
    out = nd.SoftmaxActivation(out)
    pred = int(nd.argmax(out, axis=1).asscalar())
    prob = out[0][pred].asscalar()
    label = train_imgs.synsets
    return 'With prob=%f, %s' % (prob, label[pred])


print(classify_food(finetune_net, '../data/food/test/bad/__-_xGdkEiqko_xpgxRAZQ.jpg'))
print(classify_food(finetune_net, '../data/food/test/best/__AGYSSgRK2LzIKS1cJdKQ.jpg'))

