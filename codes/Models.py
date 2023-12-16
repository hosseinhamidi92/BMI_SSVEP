from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D,Add
from tensorflow.keras.layers import Conv1DTranspose,Activation,Flatten
from tensorflow.keras.layers import LeakyReLU,LSTM,Bidirectional
from tensorflow.keras.layers import Activation,MaxPooling1D
from tensorflow.keras.layers import Concatenate,Dense,Multiply
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,GlobalMaxPooling2D,Reshape
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.constraints import max_norm
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import KFold
import math
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.losses import categorical_crossentropy


def SE_block(input_shape):
    x = GlobalMaxPooling2D()(input_shape)
    x = Dense(int(int(input_shape.shape[-1])/8),activation='relu')(x)
    x = Dense(int(input_shape.shape[-1]),activation='sigmoid')(x)
    x = Reshape((-1,int(input_shape.shape[-1])))(x)
    #x = Concatenate()([x, input_shape])
    x = Multiply()([x, input_shape])
    return x
    
   



def SE_resnet(n_filters, input_layer,filter_shape, CNN_PARAMS,res):
    
    g = Conv2D(n_filters, filter_shape, padding='same', kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']))(input_layer)
    g = BatchNormalization()(g)
    g = Activation('relu')(g)
    #g = Dropout(CNN_PARAMS['droprate'])(g)

    g = SE_block(g)
    
	# concatenate merge channel-wise with input layer
    #g = Concatenate()([g, a])
    if res:
        g = Concatenate()([g, input_layer])
    g = Activation('relu')(g)
    return g




def load_Model2(input_shape, CNN_PARAMS):
    inpt = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    x = Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(input_shape[0], 1), 
                     input_shape=(input_shape[0], input_shape[1], input_shape[2]), 
                     padding="valid", kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), 
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(inpt)
    x = BatchNormalization()(x)
    a = Activation('relu')(x)
    #a = SE_block(x)
    #a = Dropout(CNN_PARAMS['droprate'])(x)
    x = Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(1,10), 
                     kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), padding="same", 
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(a)
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Add()([x, a])
    a2 = SE_block(x)
    a2 = Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(1, 3), 
                     kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), padding="same", 
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(a2)
    a2 = BatchNormalization()(a2)
    a2 = Activation('relu')(a2)
    #x = Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(1,1), 
    #                 kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), padding="same", 
    #                 kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(a)
    
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    
    x = Add()([x,a2])
    #x = Concatenate()([x, a2])
    
    #x = SE_block(x)
    
    
    #x = keras.layers.GlobalAveragePooling2D()(x)
    x = Dropout(CNN_PARAMS['droprate'])(x)
    
    x = Flatten()(x)
    #x = Dropout(CNN_PARAMS['droprate'])(x)
    out = Dense(CNN_PARAMS['num_classes'], activation='softmax', 
                    kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), 
                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(x)
    
    model = Model(inpt, out)
    return model




def load_Model3(input_shape, CNN_PARAMS):
    inpt = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    x = Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(CNN_PARAMS['n_ch'], 1), 
                     input_shape=(input_shape[0], input_shape[1], input_shape[2]), 
                     padding="valid", kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), 
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(inpt)
    x = BatchNormalization()(x)
    a = Activation('relu')(x)
    #a = SE_block(x)
    #a = Dropout(CNN_PARAMS['droprate'])(x)
    x = Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(1,10), 
                     kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), padding="same", 
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(a)
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    #x = Add()([x, a])
    #a2 = SE_block(x)
    #a2 = Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(1, 3), 
                    # kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), padding="same", 
                     #kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(a2)
    #a2 = BatchNormalization()(a2)
    #a2 = Activation('relu')(a2)
    #x = Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(1,1), 
    #                 kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), padding="same", 
    #                 kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(a)
    
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    
    x = Add()([x,a])
    #x = Concatenate()([x, a2])
    
    #x = SE_block(x)
    
    
    #x = keras.layers.GlobalAveragePooling2D()(x)
    x = Dropout(CNN_PARAMS['droprate'])(x)
    
    x = Flatten()(x)
    #x = Dropout(CNN_PARAMS['droprate'])(x)
    out = Dense(CNN_PARAMS['num_classes'], activation='softmax', 
                    kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), 
                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(x)
    
    model = Model(inpt, out)
    return model




def load_Model4(input_shape, CNN_PARAMS):
    inpt = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    x = Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(CNN_PARAMS['n_ch'], 1), 
                     input_shape=(input_shape[0], input_shape[1], input_shape[2]), 
                     padding="valid", kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), 
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(inpt)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    a = SE_block(x)
    #a = Dropout(CNN_PARAMS['droprate'])(x)
    x = Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(1,10), 
                     kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), padding="same", 
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(a)
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    #x = Add()([x, a])
    #a2 = SE_block(x)
    #a2 = Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(1, 3), 
                    # kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), padding="same", 
                     #kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(a2)
    #a2 = BatchNormalization()(a2)
    #a2 = Activation('relu')(a2)
    #x = Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(1,1), 
    #                 kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), padding="same", 
    #                 kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(a)
    
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    
    x = Add()([x,a])
    #x = Concatenate()([x, a2])
    
    #x = SE_block(x)
    
    
    #x = keras.layers.GlobalAveragePooling2D()(x)
    x = Dropout(CNN_PARAMS['droprate'])(x)
    
    x = Flatten()(x)
    #x = Dropout(CNN_PARAMS['droprate'])(x)
    out = Dense(CNN_PARAMS['num_classes'], activation='softmax', 
                    kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), 
                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(x)
    
    model = Model(inpt, out)
    return model






from tensorflow.keras.models import Sequential
def CNN_model(input_shape, CNN_PARAMS):
    
    model = Sequential()
    model.add(Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(CNN_PARAMS['n_ch'], 1), 
                     input_shape=(input_shape[0], input_shape[1], input_shape[2]), 
                     padding="valid", kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), 
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(CNN_PARAMS['droprate']))  
    model.add(Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(1, CNN_PARAMS['kernel_f']), 
                     kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), padding="valid", 
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(CNN_PARAMS['droprate']))  
    model.add(Flatten())
    model.add(Dense(CNN_PARAMS['num_classes'], activation='softmax', 
                    kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), 
                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    
    return model


def EEGNet(input_shape, CNN_PARAMS,
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (input_shape[0], input_shape[1], input_shape[2]),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((CNN_PARAMS['n_ch'], 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = Dropout(CNN_PARAMS['droprate'])(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = Dropout(CNN_PARAMS['droprate'])(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(CNN_PARAMS['num_classes'], name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)