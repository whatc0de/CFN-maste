# -*- coding: utf-8 -*-

import numpy as np
from keras import backend as K
from keras.layers import Layer,Dense,Dropout
from keras.models import Model, Input




class Attention(Layer):

    def __init__(self, delta =0.5,c=1,dropout_keep_prob = 1, mean="geometric",axis = -1, keep_dims = True,nb_classes =2, **kwargs):
        # self.output_dim = output_dim
        self.axis = axis
        self.keep_dims = keep_dims
        self.delta = delta
        self.c = c
        self.mean=mean
        super(Attention, self).__init__(**kwargs)



    def get_config(self):
        config = {'axis': self.axis, 'keep_dims': self.keep_dims}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):

        super(Attention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        x,y = inputs
        multipled = x*y
        weight = K.softmax(multipled,axis=-1)
        
        representations = K.concatenate([x, y,multipled,multipled*weight], axis=-1)
        return representations

    def compute_output_shape(self, input_shape):
        none_batch , dim = input_shape[0]
        dim = dim * 4
        return([tuple([none_batch,dim])])


if __name__ == '__main__':
    from keras.layers import Input, Dense
    x =  Input(shape=(2,))
    y =  Input(shape=(2,))

    output = Attention()([x,y])

    encoder = Model([x,y], output)
    encoder.compile(loss = 'mean_squared_error',
            optimizer = 'rmsprop',
            metrics=['accuracy'])   
    a = np.random.random((5,300))
    b = np.random.random((5,300))
    c = np.random.random((5,1))
    a = np.ones((5,300))
    
    a= np.array([[1,1],[3,4]])
    b= np.array([[1,0],[4,3]])
    print(encoder.predict(x = [a,b]))
