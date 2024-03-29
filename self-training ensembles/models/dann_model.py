import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.models


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means)/sds

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DigitsDANNModel(nn.Module):
    def __init__(self, g_num=2, f_num=3, normalizer=None):
        super(DigitsDANNModel, self).__init__()

        self.normalize_layer = NormalizeLayer(list(normalizer.mean), list(normalizer.std))

        self.h_dim_1 = 64
        self.h_dim = 128
        self.output_dim = 256

        # Encoder
        self.feature = nn.Sequential()
        self.feature.add_module('g_conv1', nn.Conv2d(3, self.h_dim_1, kernel_size=5))
        self.feature.add_module('g_bn1', nn.BatchNorm2d(self.h_dim_1))
        self.feature.add_module('g_pool1', nn.MaxPool2d(2))
        self.feature.add_module('g_relu1', nn.ReLU(True))
        self.feature.add_module('g_conv2', nn.Conv2d(self.h_dim_1, self.h_dim, kernel_size=5))
        self.feature.add_module('g_bn2', nn.BatchNorm2d(self.h_dim))
        self.feature.add_module('g_drop1', nn.Dropout2d())
        self.feature.add_module('g_pool2', nn.MaxPool2d(2))
        self.feature.add_module('g_relu2', nn.ReLU(True))
        for i in range(g_num):
            self.feature.add_module('g_conv'+str(i+3), nn.Conv2d(self.h_dim, self.h_dim, kernel_size=3, padding=1))
            self.feature.add_module('g_bn'+str(i+3), nn.BatchNorm2d(self.h_dim))
            self.feature.add_module('g_relu'+str(i+3), nn.ReLU(True))

        # Discriminator
        self.feature_d = nn.Sequential()
        for i in range(5):
            self.feature_d.add_module('df_conv'+str(i), nn.Conv2d(self.h_dim, self.h_dim, kernel_size=3, padding=1))
            self.feature_d.add_module('df_relu'+str(i), nn.ReLU(True))
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.h_dim * 4 * 4, self.output_dim))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.output_dim, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        # Predictor
        self.feature_f = nn.Sequential()
        for i in range(f_num):
            self.feature_f.add_module('f_conv'+str(i), nn.Conv2d(self.h_dim, self.h_dim, kernel_size=3, padding=1))
            self.feature_f.add_module('f_bn'+str(i), nn.BatchNorm2d(self.h_dim))
            self.feature_f.add_module('f_relu'+str(i), nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.h_dim * 4 * 4, self.output_dim))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(self.output_dim))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(self.output_dim, 10))

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def intermediate_forward(self, input_data, alpha=0.1):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        input_data = self.normalize_layer(input_data)
        feature = self.feature(input_data)
        feature = self.feature_f(feature).reshape(-1, self.h_dim * 4 * 4)
        return feature
    
    def get_logit_output(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        input_data = self.normalize_layer(input_data)

        feature = self.feature(input_data)
        feature = self.feature_f(feature).reshape(-1, self.h_dim * 4 * 4)
        class_output = self.class_classifier(feature)
        
        return class_output

    def forward(self, input_data, alpha=0.1):
        batch_size = input_data.data.shape[0]
        input_data = input_data.expand(batch_size, 3, 28, 28)
        input_data = self.normalize_layer(input_data)

        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)

        feature_d = self.feature_d(reverse_feature).reshape(batch_size, self.h_dim * 4 * 4)
        domain_output = self.domain_classifier(feature_d)

        feature = self.feature_f(feature).reshape(batch_size, self.h_dim * 4 * 4)
        class_output = self.class_classifier(feature)

        return self.logsoftmax(class_output), self.softmax(class_output), domain_output

class CifarDANNModel(nn.Module):
    def __init__(self, normalizer):
        super(CifarDANNModel, self).__init__()

        self.normalize_layer = NormalizeLayer(list(normalizer.mean), list(normalizer.std))

        # Feature extractor params
        self.ftex_layers = 1
        self.ftex_width = 256
        # Discriminator params
        self.disc_layers = 7
        self.disc_width = 256
        # Predictor params
        self.pred_layers = 3
        self.pred_width = 256

        # Encoder
        resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
        self.feature = nn.Sequential(*list(resnet.children())[:-1])
        self.feature.add_module('flatten', nn.Flatten())
        self.feature.add_module('d_fc1', nn.Linear(512, self.ftex_width))
        self.feature.add_module('d_relu1', nn.ReLU(True))
        for i in range(self.ftex_layers - 1):
            self.feature.add_module('g_fc'+str(i+2), nn.Linear(self.ftex_width, self.ftex_width))
            self.feature.add_module('g_relu'+str(i+2), nn.ReLU(True))

        # Discriminator
        self.feature_d = nn.Sequential()
        self.feature_d.add_module('df_fc1', nn.Linear(self.ftex_width, self.disc_width))
        self.feature_d.add_module('df_relu1', nn.ReLU(True))
        for i in range(self.disc_layers - 2):
            self.feature_d.add_module('df_fc'+str(i+2), nn.Linear(self.disc_width, self.disc_width))
            self.feature_d.add_module('df_relu'+str(i+2), nn.ReLU(True))
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc', nn.Linear(self.disc_width, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        # Predictor
        self.feature_f = nn.Sequential()
        self.feature_f.add_module('f_fc1', nn.Linear(self.ftex_width, self.pred_width))
        self.feature_d.add_module('f_relu1', nn.ReLU(True))
        for i in range(self.pred_layers - 2):
            self.feature_f.add_module('f_fc'+str(i+2), nn.Linear(self.pred_width, self.pred_width))
            self.feature_f.add_module('f_relu'+str(i+2), nn.ReLU(True))
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc', nn.Linear(self.pred_width, 10))

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def intermediate_forward(self, input_data, alpha=0.1):
        batch_size = input_data.data.shape[0]
        input_data = input_data.type(torch.cuda.FloatTensor)
        input_data = self.normalize_layer(input_data)
        feature = self.feature(input_data)
        feature = self.feature_f(feature).reshape(batch_size, -1)
        return feature
    
    def get_logit_output(self, input_data):
        batch_size = input_data.data.shape[0]
        input_data = input_data.type(torch.cuda.FloatTensor)
        input_data = self.normalize_layer(input_data)
        feature = self.feature(input_data)
        feature = self.feature_f(feature).reshape(batch_size, -1)
        class_output = self.class_classifier(feature)
        
        return class_output

    def forward(self, input_data, alpha=0.1):
        batch_size = input_data.data.shape[0]
        input_data = input_data.type(torch.cuda.FloatTensor)
        input_data = self.normalize_layer(input_data)

        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)

        feature_d = self.feature_d(reverse_feature).reshape(batch_size, -1)
        domain_output = self.domain_classifier(feature_d)

        feature = self.feature_f(feature).reshape(batch_size, -1)
        class_output = self.class_classifier(feature)

        return self.logsoftmax(class_output), self.softmax(class_output), domain_output


class TextDANNModel(nn.Module):
    def __init__(self):
        super(TextDANNModel, self).__init__()

        # Feature extractor params
        self.ftex_layers = 4
        self.ftex_width = 128
        # Discriminator params
        self.disc_layers = 3
        self.disc_width = 128
        # Predictor params
        self.pred_layers = 7
        self.pred_width = 256

        # Feature extractor
        self.feature = nn.Sequential()
        self.feature.add_module('g_fc1', nn.Linear(5000, self.ftex_width))
        self.feature.add_module('g_relu1', nn.ReLU(True))
        for i in range(self.ftex_layers - 1):
            self.feature.add_module('g_fc'+str(i+2), nn.Linear(self.ftex_width, self.ftex_width))
            self.feature.add_module('g_relu'+str(i+2), nn.ReLU(True))

        # Discriminator
        self.feature_d = nn.Sequential()
        self.feature_d.add_module('df_fc1', nn.Linear(self.ftex_width, self.disc_width))
        self.feature_d.add_module('df_relu1', nn.ReLU(True))
        for i in range(self.disc_layers - 2):
            self.feature_d.add_module('df_fc'+str(i+2), nn.Linear(self.disc_width, self.disc_width))
            self.feature_d.add_module('df_relu'+str(i+2), nn.ReLU(True))
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc', nn.Linear(self.disc_width, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        # Predictor
        self.feature_f = nn.Sequential()
        self.feature_f.add_module('f_fc1', nn.Linear(self.ftex_width, self.pred_width))
        self.feature_d.add_module('f_relu1', nn.ReLU(True))
        for i in range(self.pred_layers - 2):
            self.feature_f.add_module('f_fc'+str(i+2), nn.Linear(self.pred_width, self.pred_width))
            self.feature_f.add_module('f_relu'+str(i+2), nn.ReLU(True))
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc', nn.Linear(self.pred_width, 2))

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def intermediate_forward(self, input_data, alpha=0.1):
        input_data = input_data.expand(input_data.data.shape[0], 5000)
        input_data = self.normalize_layer(input_data)
        feature = self.feature(input_data)
        feature = self.feature_f(feature)
        return feature
    
    def get_logit_output(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 5000)
        input_data = self.normalize_layer(input_data)

        feature = self.feature(input_data)
        feature = self.feature_f(feature)
        class_output = self.class_classifier(feature)
        
        return class_output

    def forward(self, input_data, alpha=0.1):
        batch_size = input_data.data.shape[0]
        input_data = input_data.expand(batch_size, 5000)

        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)

        feature_d = self.feature_d(reverse_feature)
        domain_output = self.domain_classifier(feature_d)

        feature = self.feature_f(feature)
        class_output = self.class_classifier(feature)

        return self.logsoftmax(class_output), self.softmax(class_output), domain_output
