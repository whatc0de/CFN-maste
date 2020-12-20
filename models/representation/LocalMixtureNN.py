# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalMaxPooling1D,GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation,concatenate,Reshape, Permute

from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from layers import *

import math
import numpy as np

from keras import regularizers
import keras.backend as K

from models.BasicModel import BasicModel
class LocalMixtureNN(BasicModel):

    def initialize(self):
        
        self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='float32')

        self.ngram = [NGram(n_value = int(n_value)) for n_value in self.opt.ngram_value.split(',')]
        self.phase_embedding= phase_embedding_layer(None, self.opt.lookup_table.shape[0], self.opt.lookup_table.shape[1], trainable = self.opt.embedding_trainable,l2_reg=self.opt.phase_l2)

        self.amplitude_embedding = amplitude_embedding_layer(np.transpose(self.opt.lookup_table), None, trainable = self.opt.embedding_trainable, random_init = self.opt.random_init,l2_reg=self.opt.amplitude_l2)

        self.l2_normalization = L2Normalization(axis = 3)
        self.l2_norm = L2Norm(axis = 3, keep_dims = False)

        self.weight_embedding = Embedding(self.opt.lookup_table.shape[0], 1, trainable = True)

        
        
        self.dense = Dense(self.opt.nb_classes, activation=self.opt.activation, kernel_regularizer= regularizers.l2(self.opt.dense_l2))  # activation="sigmoid",
        self.dropout_embedding = Dropout(self.opt.dropout_rate_embedding)
        self.dropout_probs = Dropout(self.opt.dropout_rate_probs)
        self.projection = ComplexMeasurement(units = self.opt.measurement_size)

    def __init__(self,opt):
        super(LocalMixtureNN, self).__init__(opt)


    def build(self):
        self.probs = self.get_representation(self.doc)
        output = self.dense(self.probs)
        model = Model(self.doc, output)
        return model
    
    def get_representation(self,doc):
        
        probs_list = []
        for n_gram in self.ngram:
            self.inputs = n_gram(doc)

            self.phase_encoded = self.phase_embedding(self.inputs)
            self.amplitude_encoded = self.amplitude_embedding(self.inputs)

            self.weight = Activation('softmax')(self.l2_norm(self.amplitude_encoded))
            self.amplitude_encoded = self.l2_normalization(self.amplitude_encoded)

            if math.fabs(self.opt.dropout_rate_embedding -1) < 1e-6:
                self.phase_encoded = self.dropout_embedding(self.phase_encoded)
                self.amplitude_encoded = self.dropout_embedding(self.amplitude_encoded)

            [seq_embedding_real, seq_embedding_imag] = ComplexMultiply()([self.phase_encoded, self.amplitude_encoded])
            if self.opt.network_type.lower() == 'complex_mixture':
                [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([seq_embedding_real, seq_embedding_imag, self.weight])

            elif self.opt.network_type.lower() == 'complex_superposition':
                [sentence_embedding_real, sentence_embedding_imag]= ComplexSuperposition()([seq_embedding_real, seq_embedding_imag, self.weight])

            else:
                
                [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([seq_embedding_real, seq_embedding_imag, self.weight])
        
            probs_list.append(self.projection([sentence_embedding_real, sentence_embedding_imag]))

        self.probs = Concatenation(axis = -1)(probs_list)
        probs_feature = []
        for one_type in self.opt.pooling_type.split(','):
            if self.opt.pooling_type == 'max':
                probs = GlobalMaxPooling1D()(self.probs)
            elif self.opt.pooling_type == 'average':
                probs = GlobalAveragePooling1D()(self.probs)
            elif self.opt.pooling_type == 'none':
                probs = Flatten()(self.probs)
            elif self.opt.pooling_type == 'max_col':
                probs = GlobalMaxPooling1D()(Permute((2,1))(self.probs))
            elif self.opt.pooling_type == 'average_col':
                probs = GlobalAveragePooling1D()(Permute((2,1))(self.probs))
            else:
                print('Wrong input pooling type -- The default flatten layer is used.')
                probs = Flatten()(self.probs)
            probs_feature.append(probs)
        
        if len(probs_feature)>1:
            probs = concatenate(probs_feature)
        else:
            probs = probs_feature[0]
        if math.fabs(self.opt.dropout_rate_probs -1) < 1e-6:
                probs = self.dropout_probs(probs)
        
        return(probs)
    


if __name__ == "__main__":
    import keras
    from keras.layers import Input, Dense, Activation, Lambda
    import numpy as np
    from keras import regularizers
    from keras.models import Model
    import sys
    from params import Params
    from dataset import qa
    import keras.backend as K
    import units
    import itertools
    from loss import *
    from units import to_array 
    from keras.utils import generic_utils
    import argparse
    import models.representation as models
    params = Params()
    config_file = 'config/local.ini'    # define dataset in the config
    params.parse_config(config_file)
    import dataset
    reader = dataset.setup(params)
    params = dataset.process_embedding(reader,params)
    from keras.layers import Embedding, GlobalMaxPooling1D,GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation,concatenate,Reshape, Permute

    from keras.models import Model, Input, model_from_json, load_model
    from keras.constraints import unit_norm
    from complexnn import *
    import math
    import numpy as np
    
    from keras import regularizers
    import keras.backend as K
    class DottableDict(dict):
        def __init__(self, *args, **kwargs):
            dict.__init__(self, *args, **kwargs)
            self.__dict__ = self
            self.allowDotting()
        def allowDotting(self, state=True):
            if state:
                self.__dict__ = self
            else:
                self.__dict__ = dict()
            
    self = DottableDict()
    self.opt = params
    self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='float32')

    self.ngram = [NGram(n_value = int(n_value)) for n_value in self.opt.ngram_value.split(',')]
    self.phase_embedding= phase_embedding_layer(self.opt.max_sequence_length, self.opt.lookup_table.shape[0], self.opt.lookup_table.shape[1], trainable = self.opt.embedding_trainable,l2_reg=self.opt.phase_l2)

    self.amplitude_embedding = amplitude_embedding_layer(np.transpose(self.opt.lookup_table), self.opt.max_sequence_length, trainable = self.opt.embedding_trainable, random_init = self.opt.random_init,l2_reg=self.opt.amplitude_l2)
    self.l2_normalization = L2Normalization(axis = 3)
    self.l2_norm = L2Norm(axis = 3, keep_dims = False)
    self.weight_embedding = Embedding(self.opt.lookup_table.shape[0], 1, trainable = True)
    self.dense = Dense(self.opt.nb_classes, activation=self.opt.activation, kernel_regularizer= regularizers.l2(self.opt.dense_l2))  # activation="sigmoid",
    self.dropout_embedding = Dropout(self.opt.dropout_rate_embedding)
    self.dropout_probs = Dropout(self.opt.dropout_rate_probs)
    self.projection = ComplexMeasurement(units = self.opt.measurement_size)

    
    doc = self.doc
