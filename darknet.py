import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def parse_cfg(filepath):
    file = open(filepath, 'r')
    lines = file.read().split('\n')
    lines = [line for line in lines if len(line) > 0]
    lines = [line for line in lines if line[0] != '#']
    lines = [line.rstrip().lstrip() for line in lines]

    block = dict()
    blocks = list()

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = dict()
            block['type'] = line[1:-1]
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = list()

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # Convolutional Layers
        if x['type'] == 'convolutional':
            activation = x['activation']

            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module('conv_{0}'.format(index), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm{0}'.format(index), bn)

            if activation == 'leaky':
                activ = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), activ)

        # Upsampling layer
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module('upsample_{0}'.format(index), upsample)

        # Route layer
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end = 0

            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module('route_{0}'.format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # Shortcut layer
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{0}'.format(index), shortcut)

        # YOLO detection layer
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(i) for i in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{0}'.format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


if __name__ == '__main__':
    blocks = parse_cfg('cfg/yolov3.cfg')
    print(create_modules(blocks))
