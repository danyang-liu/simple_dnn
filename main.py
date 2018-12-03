from metric import *
from data_process import *
from conf import *

# truth, score = get_truth_score(udpairs_dv_uv_test_raw)
#
# auc = cal_auc(truth, score)
#
# print(str(auc))

# split_train_test_by_session_id(udpairs_dv_uv, udpairs_dv_uv_session_train, udpairs_dv_uv_session_test)
sample_train(udpairs_dv_uv_session_train, udpairs_dv_uv_session_train_sample)