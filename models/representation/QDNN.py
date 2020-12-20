# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation,Multiply,Concatenate,Add,Reshape,LeakyReLU
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from layers import *
#import keras.backend as K
from keras import backend as K
import math
import numpy as np
import tensorflow as tf

from keras import regularizers



class QDNN(BasicModel):

    def initialize(self): 

        self.input_amplitude = Input(shape=(9,768,),dtype='float32')
        self.input_phase = Input(shape=(9,768,), dtype='float32')
        self.weight = Input(shape=(8,), dtype='float32')
   
        
        self.dense = Dense(self.opt.nb_classes, activation=self.opt.activation, kernel_regularizer= regularizers.l2(self.opt.dense_l2))  # activation="sigmoid",
        
        
        self.dropout_embedding = Dropout(self.opt.dropout_rate_embedding)
        
        
        self.dropout_probs = Dropout(self.opt.dropout_rate_probs)
        
        
        self.projection = ComplexMeasurement(units = self.opt.measurement_size)

    def __init__(self,opt):
        super(QDNN, self).__init__(opt)


    def build(self):
        
        probs = self.get_representation(self.input_amplitude,self.input_phase,self.weight)
        if self.opt.network_type== "ablation" and self.opt.ablation == 1:
            predictions = ComplexDense(units = self.opt.nb_classes, activation= "sigmoid", init_criterion = self.opt.init_mode)(probs)
            output = GetReal()(predictions)
        else:
            output = self.dense(probs)
        model = Model([self.input_amplitude,self.input_phase,self.weight], output)
        return model
    
    def get_representation(self,input_amplitude,input_phase,weight):
        
        self.weight = weight
        
        self.phase_encoded = input_amplitude
        self.amplitude_encoded = input_phase
        
        densesize = 600 ## 300 50 100
        
        
        self.phase_encoded = Dense(densesize, activation=self.opt.activation)(self.phase_encoded)
        self.amplitude_encoded = Dense(densesize, activation=self.opt.activation)(self.amplitude_encoded)
        
        
        
        if math.fabs(self.opt.dropout_rate_embedding -1) < 1e-6:
            self.phase_encoded = self.dropout_embedding(self.phase_encoded)
            self.amplitude_encoded = self.dropout_embedding(self.amplitude_encoded)

        x = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 9})(self.amplitude_encoded)###12 = maxcontextnum+num(utt)
        angle = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 9})(self.phase_encoded)###12 = maxcontextnum+num(utt)
        uttlayer_amplitude = x[-1]
        uttlayer_phase = angle[-1]
        concatlayers_amplitude = []
        concatlayers_phase = []
        
        for i in range(8):
            tmplayer_amplitude = []
            tmplayer_amplitude.append(uttlayer_amplitude)
            for d in range(i+1):
                tmplayer_amplitude.append(x[d])
            tmprealr_amplitude = Multiply()(tmplayer_amplitude)
            concatlayers_amplitude.append(tmprealr_amplitude)
        realr = Concatenate(axis=1)(concatlayers_amplitude)
        
        for i in range(8):
            tmplayer_phase = []
            tmplayer_phase.append(uttlayer_phase)
            for d in range(i+1):
                tmplayer_phase.append(angle[d])
            tmplayer_phase = Add()(tmplayer_phase)
            concatlayers_phase.append(tmplayer_phase)
        imaga = Concatenate(axis=1)(concatlayers_phase)
      
        cosa = Lambda(lambda x:  K.cos(x))(imaga)
        sina = Lambda(lambda x:  K.sin(x))(imaga)
        finalreal = Multiply()([cosa,realr])
        finalimag = Multiply()([sina,realr])
        
        if self.opt.network_type.lower() == 'complex_mixture':
            [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([seq_embedding_real, seq_embedding_imag, self.weight])

        elif self.opt.network_type.lower() == 'complex_superposition':
            [sentence_embedding_real, sentence_embedding_imag]= ComplexSuperposition()([seq_embedding_real, seq_embedding_imag, self.weight])

        else:
            print('Wrong input network type -- The default mixture network is constructed.')
            
            [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([finalreal, finalimag,self.weight])


        if self.opt.network_type== "ablation" and self.opt.ablation == 1:
            sentence_embedding_real = Flatten()(sentence_embedding_real)
            sentence_embedding_imag = Flatten()(sentence_embedding_imag)
            probs = [sentence_embedding_real, sentence_embedding_imag]
        else:
            
            probs =  self.projection([sentence_embedding_real, sentence_embedding_imag])
            ###ComplexMeasurement(units = self.opt.measurement_size)([sentence_embedding_real, sentence_embedding_imag])
            
            
            if math.fabs(self.opt.dropout_rate_probs -1) < 1e-6:
                probs = self.dropout_probs(probs)
            probs = self.dropout_probs(probs)
        return(probs)


