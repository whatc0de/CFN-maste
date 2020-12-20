import json
import numpy as np

def load_my_data():
    con_data = []
    for i in range(690):
        tmp_con = np.loadtxt(\
                     'C:\\Users\\lenovo\\experiment\\source_data\\bertdata\\condata\\qnn_con'+str(i)+'.txt',\
                     dtype=float, \
                     delimiter=",")
        #print(tmp_con.shape)
        con_data.append(tmp_con)
        
    utt_data = np.loadtxt(\
                     'C:\\Users\\lenovo\\experiment\\source_data\\bertdata\\uttdata\\qnn_utt.txt',\
                     dtype=float, \
                     delimiter=",")
    con_data = np.array(con_data)
    
    return con_data,utt_data

def get_my_processed_data_passed():
    conutt_data = []
    for i in range(690):
        tmp_con = np.loadtxt(\
                     'C:\\Users\\lenovo\\experiment\\source_data\\bertdata\\processedCon_Utt\\pcon'+str(i)+'.txt',\
                     dtype=float, \
                     delimiter=",")
        #print(tmp_con.shape)
        conutt_data.append(tmp_con)
        
    label = np.loadtxt(\
                     'C:\\Users\\lenovo\\experiment\\source_data\\bertdata\\label.txt',\
                     dtype=float, \
                     delimiter=",")
    train_x = np.array(conutt_data[:600])
    train_y = np.array(label[:600])
    test_x = np.array(conutt_data[600:650])
    test_y = np.array(label[600:650])
    val_x = np.array(conutt_data[650:690])
    val_y = np.array(label[650:690])
    return (train_x,train_y),(test_x,test_y),(val_x,val_y)


def get_my_processed_data():
    tmp_con = np.loadtxt(\
                 'C:\\Users\\lenovo\\experiment\\source_data\\bertdata\\reverseeddoc.txt',\
                 dtype=int, \
                 delimiter=",")
    #print(tmp_con.shape)
        
    label = np.loadtxt(\
                     'C:\\Users\\lenovo\\experiment\\source_data\\bertdata\\label.txt',\
                     dtype=float, \
                     delimiter=",")
    train_x = np.array(tmp_con[:600])
    train_y = np.array(label[:600])
    test_x = np.array(tmp_con[600:650])
    test_y = np.array(label[600:650])
    val_x = np.array(tmp_con[650:690])
    val_y = np.array(label[650:690])
    return (train_x,train_y),(test_x,test_y),(val_x,val_y)

def get_my_processed_data_cross():
    tmp_con = np.loadtxt(\
                 'C:\\Users\\lenovo\\experiment\\source_data\\bertdata\\reverseeddoc.txt',\
                 dtype=int, \
                 delimiter=",")
    #print(tmp_con.shape)
        
    label = np.loadtxt(\
                     'C:\\Users\\lenovo\\experiment\\source_data\\bertdata\\label.txt',\
                     dtype=float, \
                     delimiter=",")
    train_x = np.array(tmp_con[:650])
    train_y = np.array(label[:650])
    val_x = np.array(tmp_con[650:690])
    val_y = np.array(label[650:690])
    return (train_x,train_y),(val_x,val_y)

def get_my_processed_source_data_cross():
    
    tmp_con = []
    tmp_phase = []
    for i in range(4400):
        utt_con = np.loadtxt(\
                     'dataset2/utt_con/utt_con'+str(i)+'.txt',\
                     dtype=float, \
                     delimiter=",")
        tmp_con.append(utt_con)
        
        phase = np.loadtxt(\
                     'dataset2/phase/phase'+str(i)+'.txt',\
                     dtype=float, \
                     delimiter=",")
        tmp_phase.append(phase)
    #print(tmp_con.shape)
        
    label = np.loadtxt(\
                     'dataset2/label.txt',\
                     dtype=int, \
                     delimiter=",")
    
    
    train_phase = np.array(tmp_phase[:4000])
    val_phase = np.array(tmp_phase[4000:4400])
    train_x = np.array(tmp_con[:4000])
    val_x = np.array(tmp_con[4000:4400])
    train_y = np.array(label[:4000])
    val_y = np.array(label[4000:4400])
    return (train_x,train_phase,train_y),(val_x,val_phase,val_y)

###此方法生成的docwithindex已弃用，现在使用上面方法中的reverseeddoc.txt，为逆序版本
def get_my_lookup_table():
    utt = np.loadtxt(\
                 'C:\\Users\\lenovo\\experiment\\source_data\\bertdata\\uttdata\\qnn_utt.txt',\
                 dtype=float, \
                 delimiter=",")
    con_data = []
    con_data.append(np.zeros((768),dtype=float))
    maxlen = 0
    doc = []
    allvecnum = 1
    #####最多context数是11
    for i in range(690):
        initdocvec = np.zeros((12),dtype=int)
        tmp_con = np.loadtxt(\
                     'C:\\Users\\lenovo\\experiment\\source_data\\bertdata\\condata\\qnn_con'+str(i)+'.txt',\
                     dtype=float, \
                     delimiter=",")
        if 768 == len(tmp_con):
            tmp_con = tmp_con.reshape((1,768))
         
        t_cal = 0
        for t in tmp_con:
            con_data.append(t)
            initdocvec[t_cal] = allvecnum
            allvecnum = allvecnum+1
            t_cal = t_cal+1
            
        con_data.append(utt[i])
        initdocvec[-1] = allvecnum
        allvecnum = allvecnum+1
        
        doc.append(initdocvec)
    doc = np.asarray(doc)
    np.savetxt("../source_data/bertdata/docwithindex.txt", doc, fmt="%d", delimiter=",")
    lookup_table = np.asarray(con_data)
    np.savetxt("../source_data/bertdata/lookuptable.txt", lookup_table, fmt="%f", delimiter=",")
    return (lookup_table)

def get_my_weights():
    allWeights = np.loadtxt(\
                     'C:\\Users\\lenovo\\experiment\\source_data\\bertdata\\weights.txt',\
                     dtype=float, \
                     delimiter=",")
    #allWeights = allWeights.reshape((690,11,1))
    train_w = allWeights[:600]
    test_w = allWeights[600:650]
    val_w = allWeights[650:690]
    return train_w,test_w,val_w

def get_my_weights_cross():
    allWeights = np.loadtxt(\
                     'dataset2/weights.txt',\
                     dtype=float, \
                     delimiter=",")
    #allWeights = allWeights.reshape((690,11,1))
    train_w = allWeights[:4000]
    val_w = allWeights[4000:]
    return train_w,val_w