# -*- coding: utf-8 -*-

import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input
import sys

from copy import copy

class Concatenation(Layer):

    def __init__(self, axis = 1, **kwargs):
        # self.output_dim = output_dim
        self.axis = axis
        super(Concatenation, self).__init__(**kwargs)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Concatenation, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):
        super(Concatenation, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):



        output = K.concatenate(inputs,axis = self.axis)
        return output

    def compute_output_shape(self, input_shape):

        if self.axis<0:
            self.axis = self.axis + len(input_shape[0])
        new_dim = sum([single_shape[self.axis]  for single_shape in input_shape])
        output_shape = list(input_shape[0])
        output_shape[self.axis] = new_dim
        return [tuple(output_shape)]


def main():
    from keras.layers import Input, Dense
    encoding_dim = 50
    input_dim = 300
    a=np.random.random([5,300])
    input_img = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim)(input_img) #,
    new_code=Concatenation()(encoded)

    encoder = Model(input_img, new_code)

    b=encoder.predict(a)
    print(np.linalg.norm(b,axis=1))

if __name__ == '__main__':
    main()
