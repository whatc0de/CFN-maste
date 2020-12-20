import numpy as np
from keras import backend as K
from keras.layers import Layer,Reshape
from keras import layers as La
from keras.models import Model, Input



class ComplexMixture(Layer):

    def __init__(self, average_weights=False, **kwargs):
        self.average_weights = average_weights
        super(ComplexMixture, self).__init__(**kwargs)


    def get_config(self):
        config = {'average_weights': self.average_weights}
        base_config = super(ComplexMixture, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called '
                             'on a list of 2/3 inputs.')

        if len(input_shape) != 3 and len(input_shape) != 2 :
             raise ValueError('This layer should be called '
                             'on a list of 2/3 inputs.'
                              'Got ' + str(len(input_shape)) + ' inputs.')


        super(ComplexMixture, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2/3 inputs.')

        if len(inputs) != 3 and len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2/3 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')

        
        ndims = len(inputs[0].shape)###句子的长度
        input_real = K.expand_dims(inputs[0]) #shape: (None, 60, 300, 1)
        input_imag = K.expand_dims(inputs[1]) #shape: (None, 60, 300, 1)

        
        input_real_transpose = K.expand_dims(inputs[0], axis = ndims-1) #shape: (None, 60, 1, 300)
        input_imag_transpose = K.expand_dims(inputs[1], axis = ndims-1) #shape: (None, 60, 1, 300)


        output_real = La.multiply([input_real_transpose,input_real])+ La.multiply([input_imag_transpose, input_imag])

        output_imag = La.multiply([input_real_transpose, input_imag])-La.multiply([input_imag_transpose,input_real])
        
        if self.average_weights:
            output_r = K.mean(output_real, axis = ndims-2, keepdims = False)
            output_i = K.mean(output_imag, axis = ndims-2, keepdims = False)
        else:
            if len(inputs[2].shape) == ndims-1:  
                newweight = K.expand_dims(K.expand_dims(inputs[2]))
                newweight = K.repeat_elements(newweight, output_real.shape[-1], axis = ndims-1)
            else:
                newweight = K.expand_dims(inputs[2])
            newweight = K.repeat_elements(newweight, output_real.shape[-1], axis = ndims)
  
            output_real = output_real*newweight 
            output_r = K.sum(output_real, axis = ndims-2) 
 
            output_imag = output_imag*newweight## 11 11 100 100
            output_i = K.sum(output_imag, axis = ndims-2)## 11 100 100

        return [output_r, output_i]

    def compute_output_shape(self, input_shape):
        one_input_shape = list(input_shape[0])
        one_output_shape = []
        for i in range(len(one_input_shape)):
            if not i== len(one_input_shape)-2:
                one_output_shape.append(one_input_shape[i])
        one_output_shape.append(one_output_shape[-1])
        return [tuple(one_output_shape), tuple(one_output_shape)]


def main():
    input_2 = Input(shape=(3,5,), dtype='float')
    input_1 = Input(shape=(3,5,), dtype='float')
    weights = Input(shape = (3,), dtype = 'float')
    [output_1, output_2] = ComplexMixture( )([input_1, input_2, weights])


    model = Model([input_1, input_2, weights], [output_1, output_2])
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    model.summary()

    x = np.random.random((3,3,5))
    x_2 = np.random.random((3,3,5))
    weights = np.random.random((3,3))
    output = model.predict([x,x_2, weights])
    print(output)

if __name__ == '__main__':
    main()
