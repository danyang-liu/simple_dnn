from scipy import spatial
import numpy as np
from conf import *
import csv
from metric import *
import random

# fp_score = open(dot_cos_socre_file_small,'w')
#
# with open(udpairs_dv_uv_small) as tsvfile:
#     reader = csv.reader(tsvfile, delimiter='\t')
#     for row in reader:
#         dv = []
#         uv = []
#         vec_split1 = row[9].split()
#         for i in range(len(vec_split1)):
#             dv.append(float(vec_split1[i]))
#         vec_split2 = row[10].split()
#         for i in range(len(vec_split2)):
#             uv.append(float(vec_split2[i]))
#
#         dot_score = np.dot(np.array(dv),np.array(uv))
#         cos_score = 1 - spatial.distance.cosine(dv, uv)
#         score = float(row[11])
#         fp_score.write(str(dot_score)+'\t'+str(cos_score)+'\t'+str(score)+'\n')
#
#
# fp_score.close()
truth = []
pred = []
sess_id = []


with open("D:/v-danyal/MSN/data/exp_debug/2018-11-16_session_dv_uv_f.tsv") as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        sess_id.append(row[0])
        truth.append([float(row[3])])
        pred.append([float(row[5])])
        # pred.append([random.random()])

# auc = cal_auc(truth, pred)
#
# print(auc)

ndcg = []
truth_i = []
pred_i = []
for i in range(len(truth)):
     truth_i.append(truth[i])
     pred_i.append(pred[i])
     if i + 1 == len(truth):
          ndcg.append(cal_ndcg(truth_i, pred_i, 10))
          truth_i = []
          pred_i = []
     elif sess_id[i] != sess_id[i+1]:
          ndcg.append(cal_ndcg(truth_i, pred_i, 10))
          truth_i = []
          pred_i = []
ndcg = np.mean(np.array(ndcg))

print(ndcg)

