from conf import *
import csv
import random

def get_truth_score(file):
    truth = []
    score = []
    with open(file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            truth.append(float(row[5]))
            score.append(float(row[11]))

    return truth, score

def random_split_train_test(origin_file):
    pass

def get_dv_uv_train(file):
    dv = []
    uv = []
    truth = []
    with open(file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            truth.append(float(row[5]))
            dv.append(parse_dssm_vec(row[9]))
            uv.append(parse_dssm_vec(row[10]))
    return dv, uv, truth

def parse_dssm_vec(vec):
    vec_list = []
    vec_split = vec.split()
    for i in range(len(vec_split)):
        vec_list.append(float(vec_split[i]))
    return vec_list

def get_combine_vec(vec1, vec2):
    vec_list = []
    vec_split1 = vec1.split()
    for i in range(len(vec_split1)):
        vec_list.append(float(vec_split1[i]))
    vec_split2 = vec2.split()
    for i in range(len(vec_split2)):
        vec_list.append(float(vec_split2[i]))
    return vec_list

def build_train_simple_dnn(file):
    Data = {}
    X = []
    Y = []
    with open(file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            combine_vector = get_combine_vec(row[9], row[10])
            X.append(combine_vector)
            Y.append([float(row[5])])
    Data['X'] = X
    Data['Y'] = Y
    return Data

def build_train_simple_dnn_session(file):
    Data = {}
    X = []
    Y = []
    Sess_id = []
    with open(file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            combine_vector = get_combine_vec(row[2], row[3])
            X.append(combine_vector)
            Y.append([float(row[1])])
            Sess_id.append(row[0])
    Data['X'] = X
    Data['Y'] = Y
    Data['Sess_id'] = Sess_id
    return Data

def split_train_val_test(origin_file, train_val_file, test_file):

    fp = open(origin_file, 'r')
    fp_train_val = open(train_val_file, 'w')
    fp_test = open(test_file, 'w')

    for line in fp:
        if random.random() < 0.8:
            fp_train_val.write(line)
        else:
            fp_test.write(line)

    fp.close()
    fp_train_val.close()
    fp_test.close()

def sample_train_val(train_val_raw_file, train_val_sample_file):
    fp = open(train_val_raw_file, 'r')
    fp_sample = open(train_val_sample_file, 'w')
    for line in fp:
        linesplit = line.split()
        if (linesplit[5] == '0' and random.random() < 0.1121) or linesplit[5] == '1':
            fp_sample.write(line)
    fp.close()
    fp_sample.close()

def split_train_val(origin_file, train_file, val_file):
    fp = open(origin_file, 'r')
    fp_train = open(train_file, 'w')
    fp_val = open(val_file, 'w')

    for line in fp:
        if random.random() < 0.875:
            fp_train.write(line)
        else:
            fp_val.write(line)

    fp.close()
    fp_train.close()
    fp_val.close()

def split_train_test_by_session_id(origin_file, train_file, test_file):
    fp = open(origin_file, 'r')
    fp_train = open(train_file, 'w')
    fp_test = open(test_file, 'w')

    session_dict = {}

    for line in fp:
        row = line.split('\t')
        if row[0] not in session_dict:
            u_d_items = []
            u_d_items.append([])
            u_d_items[0].append(row[0])
            u_d_items[0].append(row[5])
            u_d_items[0].append(row[9])
            u_d_items[0].append(row[10])
            u_d_items[0].append(row[11])
            session_dict[row[0]] = u_d_items
        else:
            u_d_item = []
            u_d_item.append(row[0])
            u_d_item.append(row[5])
            u_d_item.append(row[9])
            u_d_item.append(row[10])
            u_d_item.append(row[11])
            session_dict[row[0]].append(u_d_item)


    for key in session_dict.keys():
        if random.random() < 0.8:
            u_d_pairs = session_dict[key]
            for i in range(len(u_d_pairs)):
                fp_train.write(u_d_pairs[i][0]+'\t'+u_d_pairs[i][1]+'\t'+u_d_pairs[i][2]+'\t'+u_d_pairs[i][3]+'\t'+u_d_pairs[i][4])
        else:
            u_d_pairs = session_dict[key]

            is_click = 0
            for i in range(len(u_d_pairs)):
                if u_d_pairs[i][1] == '1':
                    is_click = 1
            if is_click == 1 and len(u_d_pairs) > 5:
                for i in range(len(u_d_pairs)):
                    fp_test.write(
                        u_d_pairs[i][0] + '\t' + u_d_pairs[i][1] + '\t' + u_d_pairs[i][2] + '\t' + u_d_pairs[i][
                            3] + '\t' +
                        u_d_pairs[i][4])
 #           session_dict[row[0]] =

    # for line in fp:
    #     if random.random() < 0.8:
    #         fp_train.write(line)
    #     else:
    #         fp_test.write(line)

    fp.close()
    fp_train.close()
    fp_test.close()

def sample_train(train_val_raw_file, train_val_sample_file):
    fp = open(train_val_raw_file, 'r')
    fp_sample = open(train_val_sample_file, 'w')
    for line in fp:
        linesplit = line.split()
        if (linesplit[1] == '0' and random.random() < 0.1121) or linesplit[1] == '1':
            fp_sample.write(line)
    fp.close()
    fp_sample.close()

#split_train_test_by_session_id(udpairs_dv_uv, udpairs_dv_uv_session_train, udpairs_dv_uv_session_test)
#split_train_val_test(udpairs_dv_uv, udpairs_dv_uv_train_val_raw, udpairs_dv_uv_test_raw)
#sample_train_val(udpairs_dv_uv_train_val_raw, udpairs_dv_uv_train_val_sample)
#split_train_val(udpairs_dv_uv_train_val_sample, udpairs_dv_uv_train, udpairs_dv_uv_val)
