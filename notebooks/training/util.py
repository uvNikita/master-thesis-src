import random
import caffe

from collections import defaultdict
from caffe import layers as L
from caffe import params as P


def balance_dataset(dataset, limit=10000):
    img_pools = {label: list(imgs) for label, imgs in dataset.items()}
    balanced = defaultdict(set)
    # shuffle img_pools
    for imgs in img_pools.values():
            random.shuffle(imgs)
    
    def insert_image(img):
        for label, imgs in dataset.iteritems():
            if img in imgs:
                balanced[label].add(img)
    
    # process categories below the limit first
    for label, pool in img_pools.iteritems():
        if len(pool) > limit:
            continue
        while pool:
            img = pool.pop()
            insert_image(img)

    to_process = set(dataset.keys())
    while to_process:
        for label in to_process.copy():
            pool = img_pools[label]
            if not pool or len(balanced[label]) >= limit:
                # finished with this label
                to_process.discard(label)
                continue
            img = random.choice(pool)
            insert_image(img)
    return dict(balanced)


def split_dataset(dataset, ratio):
    img_pools = {label: list(imgs) for label, imgs in dataset.items()}
    split = defaultdict(set)
    # shuffle img_pools
    for imgs in img_pools.values():
            random.shuffle(imgs)
    
    def insert_image(img):
        for label, imgs in dataset.iteritems():
            if img in imgs:
                split[label].add(img)
    
    to_process = set(dataset.keys())
    while to_process:
        for label in to_process.copy():
            pool = img_pools[label]
            if not pool or float(len(split[label])) / len(dataset[label]) >= ratio:
                # finished with this label
                #print label, len(pool), len(balanced[label])
                to_process.discard(label)
                continue
            img = pool.pop()
            insert_image(img)
    return dict(split), {label: imgs - split[label] for label, imgs in dataset.iteritems()}


def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def add_caffenet(net, num_labels):
    net.conv1, net.relu1 = conv_relu(net.data, 11, 96, stride=4)
    net.pool1 = max_pool(net.relu1, 3, stride=2)
    net.norm1 = L.LRN(net.pool1, local_size=5, alpha=1e-4, beta=0.75)
    net.conv2, net.relu2 = conv_relu(net.norm1, 5, 256, pad=2, group=2)
    net.pool2 = max_pool(net.relu2, 3, stride=2)
    net.norm2 = L.LRN(net.pool2, local_size=5, alpha=1e-4, beta=0.75)
    net.conv3, net.relu3 = conv_relu(net.norm2, 3, 384, pad=1)
    net.conv4, net.relu4 = conv_relu(net.relu3, 3, 384, pad=1, group=2)
    net.conv5, net.relu5 = conv_relu(net.relu4, 3, 256, pad=1, group=2)
    net.pool5 = max_pool(net.relu5, 3, stride=2)
    net.fc6, net.relu6 = fc_relu(net.pool5, 4096)
    net.drop6 = L.Dropout(net.relu6, in_place=True)
    net.fc7, net.relu7 = fc_relu(net.drop6, 4096)
    net.drop7 = L.Dropout(net.relu7, in_place=True)
    net.score = L.InnerProduct(net.drop7, num_output=num_labels)
    return net

def caffenet_multilabel_sigmoid(name, num_labels, data_layer_params=None, is_test=True):
    # setup the ntb data layer
    net = caffe.NetSpec()
    assert is_test or data_layer_params
    
    if is_test:
        net.data = L.Input(input_param={"shape": {"dim": [1, 3, 227, 227]}})
    else:
        net.data, net.label = L.Python(
            module='ntb.layer.data', layer='NTBDataLayer', 
            ntop=2, param_str=str(data_layer_params)
        )

    net = add_caffenet(net, num_labels)
    
    # Changed loss function
    if is_test:
        net.prob = L.Sigmoid(net.fc7)
    else:
        net.loss = L.SigmoidCrossEntropyLoss(net.score, net.label)
    
    name_field = 'name: "{}"'.format(name)
    return name_field + '\n' + str(net.to_proto())


def caffenet(lmdb, batch_size=256, include_acc=False):
    data, label = L.Data(source=lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
        transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=True))

    # the net itself
    conv1, relu1 = conv_relu(data, 11, 96, stride=4)
    pool1 = max_pool(relu1, 3, stride=2)
    norm1 = L.LRN(pool1, local_size=5, alpha=1e-4, beta=0.75)
    conv2, relu2 = conv_relu(norm1, 5, 256, pad=2, group=2)
    pool2 = max_pool(relu2, 3, stride=2)
    norm2 = L.LRN(pool2, local_size=5, alpha=1e-4, beta=0.75)
    conv3, relu3 = conv_relu(norm2, 3, 384, pad=1)
    conv4, relu4 = conv_relu(relu3, 3, 384, pad=1, group=2)
    conv5, relu5 = conv_relu(relu4, 3, 256, pad=1, group=2)
    pool5 = max_pool(relu5, 3, stride=2)
    fc6, relu6 = fc_relu(pool5, 4096)
    drop6 = L.Dropout(relu6, in_place=True)
    fc7, relu7 = fc_relu(drop6, 4096)
    drop7 = L.Dropout(relu7, in_place=True)
    fc8 = L.InnerProduct(drop7, num_output=1000)
    loss = L.SoftmaxWithLoss(fc8, label)

    if include_acc:
        acc = L.Accuracy(fc8, label)
        return to_proto(loss, acc)
    else:
        return to_proto(loss)