import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
from keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
from keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU
from keras.applications import VGG16, MobileNetV2, ResNet50

class Architectures:
    # AlexNet Architecture
    def alexNet(self, input_shape, n_classes):
        input = Input(input_shape)

        # actually batch normalization didn't exist back then
        # they used LRN (Local Response Normalization) for regularization
        x = Conv2D(96, 11, strides=4, padding='same', activation='relu')(input)
        x = BatchNormalization()(x)
        x = MaxPool2D(3, strides=2)(x)

        x = Conv2D(256, 5, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(3, strides=2)(x)

        x = Conv2D(384, 3, strides=1, padding='same', activation='relu')(x)

        x = Conv2D(384, 3, strides=1, padding='same', activation='relu')(x)

        x = Conv2D(256, 3, strides=1, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(3, strides=2)(x)

        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)

        output = Dense(n_classes, activation='softmax')(x)

        model = Model(input, output)
        return model

    # VGG Architecture
    def vgg(self, input_shape, n_classes):
        input = Input(input_shape)

        x = Conv2D(64, 3, padding='same', activation='relu')(input)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = MaxPool2D(2, strides=2, padding='same')(x)

        x = Conv2D(128, 3, padding='same', activation='relu')(x)
        x = Conv2D(128, 3, padding='same', activation='relu')(x)
        x = MaxPool2D(2, strides=2, padding='same')(x)

        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = MaxPool2D(2, strides=2, padding='same')(x)

        x = Conv2D(512, 3, padding='same', activation='relu')(x)
        x = Conv2D(512, 3, padding='same', activation='relu')(x)
        x = Conv2D(512, 3, padding='same', activation='relu')(x)
        x = MaxPool2D(2, strides=2, padding='same')(x)

        x = Conv2D(512, 3, padding='same', activation='relu')(x)
        x = Conv2D(512, 3, padding='same', activation='relu')(x)
        x = Conv2D(512, 3, padding='same', activation='relu')(x)
        x = MaxPool2D(2, strides=2, padding='same')(x)

        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)

        output = Dense(n_classes, activation='softmax')(x)

        model = Model(input, output)
        return model

    # Inception Architecture
    def googlenet(self, input_shape, n_classes):
        def inception_block(x, f):
            t1 = Conv2D(f[0], 1, activation='relu')(x)

            t2 = Conv2D(f[1], 1, activation='relu')(x)
            t2 = Conv2D(f[2], 3, padding='same', activation='relu')(t2)

            t3 = Conv2D(f[3], 1, activation='relu')(x)
            t3 = Conv2D(f[4], 5, padding='same', activation='relu')(t3)

            t4 = MaxPool2D(3, 1, padding='same')(x)
            t4 = Conv2D(f[5], 1, activation='relu')(t4)

            output = Concatenate()([t1, t2, t3, t4])
            return output

        input = Input(input_shape)

        x = Conv2D(64, 7, strides=2, padding='same', activation='relu')(input)
        x = MaxPool2D(3, strides=2, padding='same')(x)

        x = Conv2D(64, 1, activation='relu')(x)
        x = Conv2D(192, 3, padding='same', activation='relu')(x)
        x = MaxPool2D(3, strides=2)(x)

        x = inception_block(x, [64, 96, 128, 16, 32, 32])
        x = inception_block(x, [128, 128, 192, 32, 96, 64])
        x = MaxPool2D(3, strides=2, padding='same')(x)

        x = inception_block(x, [192, 96, 208, 16, 48, 64])
        x = inception_block(x, [160, 112, 224, 24, 64, 64])
        x = inception_block(x, [128, 128, 256, 24, 64, 64])
        x = inception_block(x, [112, 144, 288, 32, 64, 64])
        x = inception_block(x, [256, 160, 320, 32, 128, 128])
        x = MaxPool2D(3, strides=2, padding='same')(x)

        x = inception_block(x, [256, 160, 320, 32, 128, 128])
        x = inception_block(x, [384, 192, 384, 48, 128, 128])

        x = AvgPool2D(7, strides=1)(x)
        x = Dropout(0.4)(x)

        x = Flatten()(x)
        output = Dense(n_classes, activation='softmax')(x)

        model = Model(input, output)
        return model

    # MobileNet (V2) Architecture
    def mobilenet(self, input_shape, n_classes):
        def mobilenet_block(x, f, s=1):
            x = DepthwiseConv2D(3, strides=s, padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D(f, 1, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x

        input = Input(input_shape)

        x = Conv2D(32, 3, strides=2, padding='same')(input)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = mobilenet_block(x, 64)
        x = mobilenet_block(x, 128, 2)
        x = mobilenet_block(x, 128)

        x = mobilenet_block(x, 256, 2)
        x = mobilenet_block(x, 256)

        x = mobilenet_block(x, 512, 2)
        for _ in range(5):
            x = mobilenet_block(x, 512)

        x = mobilenet_block(x, 1024, 2)
        x = mobilenet_block(x, 1024)

        x = GlobalAvgPool2D()(x)

        output = Dense(n_classes, activation='softmax')(x)

        model = Model(input, output)
        return model

    # ResNet (50 Layers) Architecture
    def resnet(self, input_shape, n_classes):

        def conv_bn_rl(x, f, k=1, s=1, p='same'):
            x = Conv2D(f, k, strides=s, padding=p)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x

        def identity_block(tensor, f):
            x = conv_bn_rl(tensor, f)
            x = conv_bn_rl(x, f, 3)
            x = Conv2D(4 * f, 1)(x)
            x = BatchNormalization()(x)

            x = Add()([x, tensor])
            output = ReLU()(x)
            return output

        def conv_block(tensor, f, s):
            x = conv_bn_rl(tensor, f)
            x = conv_bn_rl(x, f, 3, s)
            x = Conv2D(4 * f, 1)(x)
            x = BatchNormalization()(x)

            shortcut = Conv2D(4 * f, 1, strides=s)(tensor)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            output = ReLU()(x)
            return output

        def resnet_block(x, f, r, s=2):
            x = conv_block(x, f, s)
            for _ in range(r - 1):
                x = identity_block(x, f)
            return x

        input = Input(input_shape)

        x = conv_bn_rl(input, 64, 7, 2)
        x = MaxPool2D(3, strides=2, padding='same')(x)

        x = resnet_block(x, 64, 3, 1)
        x = resnet_block(x, 128, 4)
        x = resnet_block(x, 256, 6)
        x = resnet_block(x, 512, 3)

        x = GlobalAvgPool2D()(x)

        output = Dense(n_classes, activation='softmax')(x)

        model = Model(input, output)
        return model

    # YOLO Architecture
    def yolo(self, input_shape=(448, 448, 3), n_outputs=30):
        activation = LeakyReLU(0.1)

        def conv_1_3(x, f1, f2, r=1):
            for _ in range(r):
                x = Conv2D(f1, 1, padding='same', activation=activation)(x)
                x = Conv2D(f2, 3, padding='same', activation=activation)(x)
            return x

        input = Input(input_shape)

        x = Conv2D(64, 7, strides=2, padding='same', activation=activation)(input)
        x = MaxPool2D(2, strides=2, padding='same')(x)

        x = Conv2D(192, 3, padding='same', activation=activation)(x)
        x = MaxPool2D(2, strides=2, padding='same')(x)

        x = conv_1_3(x, 128, 256)
        x = conv_1_3(x, 256, 512)
        x = MaxPool2D(2, strides=2, padding='same')(x)

        x = conv_1_3(x, 256, 512, 4)
        x = conv_1_3(x, 512, 1024)
        x = MaxPool2D(2, strides=2, padding='same')(x)

        x = conv_1_3(x, 512, 1024, 2)
        x = Conv2D(1024, 3, padding='same', activation=activation)(x)
        x = Conv2D(1024, 3, strides=2, padding='same', activation=activation)(x)

        x = Conv2D(1024, 3, padding='same', activation=activation)(x)
        x = Conv2D(1024, 3, padding='same', activation=activation)(x)

        x = Dense(4096, activation=activation)(x)
        output = Dense(n_outputs)(x)

        model = Model(input, output)
        return model

class PreTrainedModels:

    def models(self, n_classes, model='vgg', optimizer='adam', fine_tune=0):
        if model == 'resnet':
            model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        elif model == 'mobilenet':
            model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        elif model == 'vgg':
            model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

        # Defines how many layers to freeze during training.
        # Layers in the convolutional base are switched from trainable to non-trainable
        # depending on the size of the fine-tuning parameter.
        if fine_tune > 0:
            for layer in model.layers[:-fine_tune]:
                layer.trainable = False
        else:
            for layer in model.layers:
                layer.trainable = False
    # def resNet(self, n_classes, optimizer='adam', fine_tune=0):
    #     model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    #     # Defines how many layers to freeze during training.
    #     # Layers in the convolutional base are switched from trainable to non-trainable
    #     # depending on the size of the fine-tuning parameter.
    #     if fine_tune > 0:
    #         for layer in conv_base.layers[:-fine_tune]:
    #             layer.trainable = False
    #     else:
    #         for layer in conv_base.layers:
    #             layer.trainable = False
    #
    # def mobileNet(self, n_classes, optimizer='adam', fine_tune=0):
    #     model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    #     # Defines how many layers to freeze during training.
    #     # Layers in the convolutional base are switched from trainable to non-trainable
    #     # depending on the size of the fine-tuning parameter.
    #     if fine_tune > 0:
    #         for layer in conv_base.layers[:-fine_tune]:
    #             layer.trainable = False
    #     else:
    #         for layer in conv_base.layers:
    #             layer.trainable = False
    #
    # def vgg(self, n_classes, optimizer='adam', fine_tune=0):
    #     model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    #     # Defines how many layers to freeze during training.
    #     # Layers in the convolutional base are switched from trainable to non-trainable
    #     # depending on the size of the fine-tuning parameter.
    #     if fine_tune > 0:
    #         for layer in conv_base.layers[:-fine_tune]:
    #             layer.trainable = False
    #     else:
    #         for layer in conv_base.layers:
    #             layer.trainable = False