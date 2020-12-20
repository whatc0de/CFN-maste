# -*- coding: utf-8 -*-
from params import Params
from models import representation as models
from dataset import classification as dataset
from tools import units
from tools.save import save_experiment
from loadmydata import *
import itertools
import argparse
import keras.backend as K
from keras.callbacks import Callback,ReduceLROnPlateau
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import logging
import time

gpu_count = len(units.get_available_gpus())
dir_path,global_logger = units.getLogger()


class Metrics(Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data
 
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict([self.validation_data[0], self.validation_data[1], self.validation_data[2]]), -1)
        val_targ = self.validation_data[3]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        tn, fp, fn, tp = confusion_matrix(val_targ, val_predict).ravel()
        _specificity = tn / (tn+fp)
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        _val_acc = accuracy_score(val_targ, val_predict) 
        logs['val_acc'] = _val_acc
        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        logs['val_specificity'] = _specificity
        print("- val_acc: %f  — val_f1: %f — val_precision: %f — val_recall: %f — val_specificity: %f" % (_val_acc, _val_f1, _val_precision, _val_recall, _specificity))
        logger.info("- val_acc: %f  — val_f1: %f — val_precision: %f — val_recall: %f — val_specificity: %f" % (_val_acc, _val_f1, _val_precision, _val_recall, _specificity))

        return



def run(params,reader):

    
    params=dataset.process_embedding(reader,params)
    qdnn = models.setup(params)
    model = qdnn.getModel() 
    
    model.compile(loss = params.loss,optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr), metrics=['accuracy'])
    model.summary()

    (train_x, train_phase, train_y),(val_x, val_phase, val_y) = get_my_processed_source_data_cross()

    train_w, val_w = get_my_weights_cross()

    history = model.fit(x=[train_x, train_phase,train_w], \
                        y = train_y, batch_size = params.batch_size,\
                        epochs= params.epochs,\
                        validation_split=0.2,\
                       callbacks=[Metrics(valid_data=(val_x, val_phase,val_w, val_y))])
    evaluation = model.evaluate(x = [val_x, val_phase,val_w], y = val_y)

    
    return history,evaluation

grid_parameters ={
        "loss": ["categorical_crossentropy"],#"mean_squared_error"],,"categorical_hinge"
        "optimizer":["rmsprop"], #"adagrad","adamax","nadam"],,"adadelta","adam" "rmsprop"-default
        "batch_size":[64],#16,32 64
        "activation":["sigmoid"],###relu
        "amplitude_l2":[0.0000005], #0.0000005,0.0000001,
        "phase_l2":[0.0000005],
        "dense_l2":[0.00001],#0.0001,0.00001,0],
        "measurement_size" :[1500],#,50 100 300],
        "lr" : [0.001],#,0.001 0.0001 0.01
        "epochs" : [250],
        "dropout_rate_embedding" : [0.5],#0.5,0.75,0.8,0.9,1],
        "dropout_rate_probs" : [0.7],#,0.5,0.75,0.8,1]    ,
        "ablation" : [1],
#        "network_type" : ["ablation"]
    }
if __name__=="__main__":

    ##logging set
    logging.basicConfig(filemode='w')
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("log1218.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("*********************************************Start print log*********************************************")
    ##logging set

    parser = argparse.ArgumentParser(description='running the complex embedding network')
    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=gpu_count)
    parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the gpu num.',default=0)
    args = parser.parse_args()
    
    parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]
     
    parameters= parameters[::-1]
    print('experiment/qnn/run_classification.py')
    print(parameters)

    logger.info("parameters: %s",parameters)

    params = Params()
    config_file = 'config/qdnn.ini'    # define dataset in the config
    params.parse_config(config_file)    
    for parameter in parameters:
        old_dataset = params.dataset_name
        params.setup(zip(grid_parameters.keys(),parameter))
        if old_dataset != params.dataset_name:
            print("switch {} to {}".format(old_dataset,params.dataset_name))
            reader=dataset.setup(params)
            params.reader = reader
    
        history, eval = run(params,reader)
        print(eval)
        print(type(eval))
        logger.info("eval:%s",eval)
        K.clear_session()


    logger.info("*********************************************Finish*********************************************")


