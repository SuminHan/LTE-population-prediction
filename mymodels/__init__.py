from mymodels.basic import *
from mymodels.lstm import *
from mymodels.fclstm import *
from mymodels.gman import *
from mymodels.temporal_transformer import *
# from mymodels.rawlstm import *
# from mymodels.convlstm import *
# from mymodels.conv_and_lstm import *
# from mymodels.lstm_stoken import *
# from mymodels.cnn_stacks import *
# from mymodels.random_walk import *
# from mymodels.random_walk_plus import *

import sys

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)