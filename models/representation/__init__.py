from models.representation.RealNN import RealNN
from models.representation.QDNN import QDNN
#from models.representation.QDNN-Copy1 import QDNN
from models.representation.ComplexNN import ComplexNN
from models.representation.QDNNAblation import QDNNAblation
from models.representation.LocalMixtureNN import LocalMixtureNN

def setup(opt):
    print("representation network type: " + opt.network_type)
    if opt.network_type == "real":
        model = RealNN(opt)
    elif opt.network_type == "qdnn":
        #####分类选择的是这个
        model = QDNN(opt)
        #####
    elif opt.network_type == "complex":
        model = ComplexNN(opt)
    elif opt.network_type == "local_mixture":
        ###############使用的是此model，由此进入LocalMixtureNN
        model = LocalMixtureNN(opt)
        ###############
    elif opt.network_type == "ablation":
        print("run ablation")
        model = QDNNAblation(opt)
    else:
        raise Exception("model not supported: {}".format(opt.network_type))
    return model
