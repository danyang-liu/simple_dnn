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

def get_combine_vec2(vec1, vec2, vec3, vec4):
    vec_list = []
    vec_split1 = vec1.split()
    for i in range(len(vec_split1)):
        vec_list.append(float(vec_split1[i]))
    vec_split2 = vec2.split()
    for i in range(len(vec_split2)):
        vec_list.append(float(vec_split2[i]))
    vec_split3 = vec3.split()
    for i in range(len(vec_split3)):
        vec_list.append(float(vec_split3[i]))
    vec_split4 = vec4.split()
    for i in range(len(vec_split4)):
        vec_list.append(float(vec_split4[i]))
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
            combine_vector = get_combine_vec(row[6], row[7])
            X.append(combine_vector)
            Y.append([float(row[3])])
            Sess_id.append(row[0])
    Data['X'] = X
    Data['Y'] = Y
    Data['Sess_id'] = Sess_id
    return Data

# def build_train_simple_dnn_session(file):
#     Data = {}
#     X = []
#     Y = []
#     Sess_id = []
#     with open(file) as tsvfile:
#         reader = csv.reader(tsvfile, delimiter='\t')
#         for row in reader:
#             combine_vector = get_combine_vec(row[9], row[10])
#             X.append(combine_vector)
#             Y.append([float(row[3])])
#             Sess_id.append(row[0])
#     Data['X'] = X
#     Data['Y'] = Y
#     Data['Sess_id'] = Sess_id
#     return Data

# def build_train_simple_dnn_session(file):
#     Data = {}
#     X = []
#     Y = []
#     Sess_id = []
#     with open(file) as tsvfile:
#         reader = csv.reader(tsvfile, delimiter='\t')
#         for row in reader:
#             combine_vector = get_combine_vec2(row[6], row[7], row[9], row[10])
#             X.append(combine_vector)
#             Y.append([float(row[3])])
#             Sess_id.append(row[0])
#     Data['X'] = X
#     Data['Y'] = Y
#     Data['Sess_id'] = Sess_id
#     return Data

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

    fp.close()
    fp_train.close()
    fp_test.close()

def split_train_test_by_session_id_remove(origin_file, train_file, test_file):
    fp = open(origin_file, 'r')
    fp_train = open(train_file, 'w')
    fp_test = open(test_file, 'w')

    session_dict = {}

    for line in fp:
        row = line.split('\t')
        if row[0] not in session_dict:
            u_d_items = []
            u_d_items.append([])
            u_d_items[0].append(row[0]) #session_id
            u_d_items[0].append(row[1]) #user_id
            u_d_items[0].append(row[4]) #doc_id
            u_d_items[0].append(row[5]) #isclick
            u_d_items[0].append(row[7]) #start_time
            u_d_items[0].append(row[9])  #score
            session_dict[row[0]] = u_d_items
        else:
            u_d_item = []
            u_d_item.append(row[0])  # session_id
            u_d_item.append(row[1])  # user_id
            u_d_item.append(row[4])  # doc_id
            u_d_item.append(row[5])  # isclick
            u_d_item.append(row[7])  # start_time
            u_d_item.append(row[9])  # score
            session_dict[row[0]].append(u_d_item)


    for key in session_dict.keys():
        if random.random() < 0.8:
            u_d_pairs = session_dict[key]
            for i in range(len(u_d_pairs)):
                fp_train.write(u_d_pairs[i][0]+'\t'+u_d_pairs[i][1]+'\t'+u_d_pairs[i][2]+'\t'+u_d_pairs[i][3]+'\t'+u_d_pairs[i][4]+'\t'+u_d_pairs[i][5])
        else:
            u_d_pairs = session_dict[key]

            is_click = 0
            for i in range(len(u_d_pairs)):
                if u_d_pairs[i][3] == '1':
                    is_click = 1
            if is_click == 1 and len(u_d_pairs) > 5:
                for i in range(len(u_d_pairs)):
                    fp_test.write(
                        u_d_pairs[i][0] + '\t' + u_d_pairs[i][1] + '\t' + u_d_pairs[i][2] + '\t' + u_d_pairs[i][
                            3] + '\t' +
                        u_d_pairs[i][4]+'\t'+u_d_pairs[i][5])

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


def remove_more_than_ten(infile, outfile):
    fp_in = open(infile, 'r', encoding="utf8")
    fp_out = open(outfile, 'w')
    for line in fp_in:
        linesplit = line.split('\t')
        title = linesplit[2].split('#TS#')
        if len(title) > 10:
            title = title[:10]
        fp_out.write(linesplit[0]+'\t')
        fp_out.write(title[0])
        for i in range(1,len(title)):
            fp_out.write("#TS#"+title[i])
        fp_out.write('\n')
    fp_in.close()
    fp_out.close()


#as train
def sample_11_15(infile, outfile):
    fp_in = open(infile, 'r', encoding="utf8")
    fp_out = open(outfile, 'w')
    for line in fp_in:
        linesplit = line.split('\t')
        if linesplit[3] == '0' and  random.random()<0.1121:
            fp_out.write(line)
        if linesplit[3] == '1':
            fp_out.write(line)
    fp_in.close()
    fp_out.close()

#as test
def session_11_16(infile, outfile):
    fp_in = open(infile, 'r')
    fp_out = open(outfile, 'w')

    session_dict = {}
    for line in fp_in:
        row = line.split('\t')
        if row[0] not in session_dict:
            u_d_items = []
            u_d_items.append([])
            u_d_items[0].append(row[0]) #session_id
            u_d_items[0].append(row[1]) #user_id
            u_d_items[0].append(row[2]) #doc_id
            u_d_items[0].append(row[3]) #isclick
            u_d_items[0].append(row[4]) #start_time
            u_d_items[0].append(row[5])  #score
            u_d_items[0].append(row[6])  # dv
            u_d_items[0].append(row[7])  # uv
            session_dict[row[0]] = u_d_items
        else:
            u_d_item = []
            u_d_item.append(row[0])  # session_id
            u_d_item.append(row[1])  # user_id
            u_d_item.append(row[2])  # doc_id
            u_d_item.append(row[3])  # isclick
            u_d_item.append(row[4])  # start_time
            u_d_item.append(row[5])  # score
            u_d_item.append(row[6])  # dv
            u_d_item.append(row[7])  # uv
            session_dict[row[0]].append(u_d_item)

    for key in session_dict.keys():
        u_d_pairs = session_dict[key]

        is_click = 0
        for i in range(len(u_d_pairs)):
            if u_d_pairs[i][3] == '1':
                is_click = 1
        if is_click == 1 and len(u_d_pairs) > 5:
            for i in range(len(u_d_pairs)):
                fp_out.write(
                    u_d_pairs[i][0] + '\t' + u_d_pairs[i][1] + '\t' + u_d_pairs[i][2] + '\t' + u_d_pairs[i][
                        3] + '\t' +
                    u_d_pairs[i][4] + '\t' + u_d_pairs[i][5] + '\t' + u_d_pairs[i][6] + '\t' + u_d_pairs[i][7])

def get_title_of_train_test(infile, outfile):
    fp_in = open(infile, 'r', encoding="utf8")
    fp_out = open(outfile, 'w')
    for line in fp_in:
        linesplit = line.split('\t')
        fp_out.write(linesplit[8])
    fp_in.close()
    fp_out.close()


def combain_cdssm_for_train_test(infile, cdssm_file, outfile):
    fp_in = open(infile, 'r', encoding="utf8")
    fp_cdssm = open(cdssm_file, 'r', encoding="utf8")
    fp_out = open(outfile, 'w')

    session_id = []
    user_id = []
    doc_id = []
    isclick = []
    start_time = []
    score = []

    dv = []
    uv = []
    title = []
    cdssm = []

    for line in fp_in:
        linesplit = line.split('\t')
        session_id.append(linesplit[0])
        user_id.append(linesplit[1])
        doc_id.append(linesplit[2])
        isclick.append(linesplit[3])
        start_time.append(linesplit[4])
        score.append(linesplit[5])
        dv.append(linesplit[6])
        uv.append(linesplit[7])
        title.append(linesplit[8].split('\n')[0])
        pass

    for line in fp_cdssm:
        linesplit = line.split('\t')
        cdssm.append(linesplit[1])
        pass

    for i in range(len(session_id)):
        fp_out.write(session_id[i]+'\t'+user_id[i]+'\t'+doc_id[i]+'\t'+isclick[i]+'\t'+start_time[i]+'\t'+score[i]+'\t'+dv[i]+'\t'+uv[i]+'\t'+title[i]+'\t'+cdssm[i])

    fp_in.close()
    fp_cdssm.close()
    fp_out.close()

def compute_avarage_user_cdssm(infile):
    fp_in = open(infile, 'r', encoding="utf8")
    index = 0.0
    total_cdssm = []
    for i in range(128):
        total_cdssm.append(0.0)
    for line in fp_in:
        index = index + 1.0
        linesplit = line.split('\t')
        cdssm = linesplit[1].split()
        for i in range(128):
            total_cdssm[i] = total_cdssm[i] + float(cdssm[i])
    for i in range(128):
        total_cdssm[i] = total_cdssm[i] / index

    for i in range(128):
        print(str(total_cdssm[i])+' ')

def add_default_uservec_for_train_test(infile, outfile):
    # default_vec = "-0.09516152131571354 0.0384350665115715 0.01833993799559809 -0.06189119996509319 -0.09836404358458449 -0.010023427852325574 0.014098731294482914 -0.011715580576204403 0.0007548985131076964 -0.018677718813208833 -0.040274443783028385 0.06578546358605755 0.01008425193493867 0.005719725826786996 0.014262648815995337 0.022522565740729972 -0.08282694750587644 -0.004992473266280907 0.017949657184233556 -0.05824675102955573 0.027613581966948856 -0.04909527768463229 -0.03959092445560373 -0.02517870605381484 0.049151717717803925 0.03207337372068133 0.03460842919235907 0.10993482804698419 -0.03873015637198483 -0.015623815347735473 -0.04829119426798669 0.04450706845879905 0.014616708380674115 -0.01584337498464272 0.02354191262408043 0.059254291911533516 -0.033486537916733485 -0.022388328769365924 -0.030794534686668623 0.114745457936763 -0.034387978851008756 -0.023248379316873402 0.03803614331502521 -0.005953406634480647 0.061714348910330445 -0.052460245858891474 0.08641051517588309 -0.022189258247486767 -0.017626656941422618 0.09430553123975847 -0.04379205502943579 -0.02465432115175256 -0.02327140209677811 0.040623811974223385 -0.001742218997943268 0.0595399693685699 0.03296675351578557 0.048939950203852715 -0.05419358400447741 0.08739194058267932 0.009683725560545134 -0.018843441461687403 0.05820706343094846 0.019581213700732177 0.020089756936856648 0.015328820065488555 0.03905443737724043 -0.03849019233813178 -0.05385225817990051 -0.06261549747592147 0.09627598450045216 0.04247728703245788 0.03927005945934226 0.01790574825507988 0.021729506735446474 -0.006527544085457476 0.05047294587415449 -0.06239602790281755 -0.0067262719132822795 -0.02771800290248719 -0.017966764772291675 0.049462143516325656 0.07887781255179782 0.04473473477096047 0.034202548658403456 -0.07997892001313249 0.032553416105126805 -0.026653564275883848 0.0014531745821138406 0.04107472241924881 -0.011746227815986424 0.08398943296538731 -0.02663175143240732 -0.03337342541021062 0.007888037000670336 -0.017953217678003723 0.024946164707198213 0.0285992017279852 -0.06015771667755645 -0.041606592259455 -9.762445971302859e-05 -0.0363833123136055 0.0391423877408163 -0.024064627445104368 0.023255901577491873 0.0011156911926173488 -0.02731624766974321 0.04024128927289132 -0.012908882510738018 0.05239693360906632 -0.05124477106859053 -0.042855128679751955 0.046491782077939324 0.0065110122894659565 -0.03184013593233517 -0.043840712337447324 -0.008619927069605545 -0.012091942581609732 0.012693536474635923 -0.04144871327454269 0.04053453038808859 -0.06320488173317612 -0.032742288130408005 -0.07140730512003517 -0.012542799439886072 -0.02281133065129975 0.03591124664042024 -0.059211495090477205"
    # default_vec_split = default_vec.split(' ')
    # default_vec2 = ""
    # default_vec2 = default_vec2 + ("%.8f" % float(default_vec_split[0]))
    # for i in range(1,128):
    #     default_vec2 = default_vec2+' '+ ("%.8f" % float(default_vec_split[i]))
    fp_in = open(infile, 'r', encoding="utf8", errors='ignore')
    fp_out = open(outfile, 'w')
    for line in fp_in:
        linesplit = line.split('\t')
        if linesplit[10] != '\n':
        #     tmp = linesplit[0]+'\t'+linesplit[1]+'\t'+linesplit[2]+'\t'+linesplit[3]+'\t'+default_vec2+'\n'
        #     fp_out.write(tmp)
        # else:
            fp_out.write(line)

    fp_in.close()
    fp_out.close()

def deal_with_test(infile, outfile):
    fp_in = open(infile, 'r')
    fp_out = open(outfile, 'w')

    session_dict = {}
    for line in fp_in:
        row = line.split('\t')
        if row[0] not in session_dict:
            u_d_items = []
            u_d_items.append([])
            u_d_items[0].append(row[0])  # session_id
            u_d_items[0].append(row[1])  # user_id
            u_d_items[0].append(row[2])  # doc_id
            u_d_items[0].append(row[3])  # isclick
            u_d_items[0].append(row[4])  # start_time
            u_d_items[0].append(row[5])  # session_id
            u_d_items[0].append(row[6])  # user_id
            u_d_items[0].append(row[7])  # doc_id
            # u_d_items[0].append(row[8])  # isclick
            # u_d_items[0].append(row[9])  # start_time
            # u_d_items[0].append(row[10])  # start_time
            session_dict[row[0]] = u_d_items
        else:
            u_d_item = []
            u_d_item.append(row[0])  # session_id
            u_d_item.append(row[1])  # user_id
            u_d_item.append(row[2])  # doc_id
            u_d_item.append(row[3])  # isclick
            u_d_item.append(row[4])  # start_time
            u_d_item.append(row[5])  # session_id
            u_d_item.append(row[6])  # user_id
            u_d_item.append(row[7])  # doc_id
            # u_d_item.append(row[8])  # isclick
            # u_d_item.append(row[9])  # start_time
            # u_d_item.append(row[10])  # start_time
            session_dict[row[0]].append(u_d_item)

    for key in session_dict.keys():
        u_d_pairs = session_dict[key]

        is_click = 0
        for i in range(len(u_d_pairs)):
            if u_d_pairs[i][3] == '1':
                is_click = 1
        if is_click == 1 and len(u_d_pairs) > 5:
            for i in range(len(u_d_pairs)):
                fp_out.write(
                    u_d_pairs[i][0] + '\t' + u_d_pairs[i][1] + '\t' + u_d_pairs[i][2] + '\t' + u_d_pairs[i][
                        3] + '\t' +
                    u_d_pairs[i][4] + '\t' + u_d_pairs[i][5] + '\t' + u_d_pairs[i][6] + '\t' + u_d_pairs[i][7] )
                    # + '\t' + u_d_pairs[i][
                    #     8] + '\t' +
                    # u_d_pairs[i][9]+ '\t' +
                    # u_d_pairs[i][10])

def split_test(infile, outfile1, outfile2):
    fp_in = open(infile, 'r')
    fp_out1 = open(outfile1, 'w')
    fp_out2 = open(outfile2, 'w')

    session_dict = {}
    for line in fp_in:
        row = line.split('\t')
        if row[0] not in session_dict:
            u_d_items = []
            u_d_items.append([])
            u_d_items[0].append(row[0])  # session_id
            u_d_items[0].append(row[1])  # user_id
            u_d_items[0].append(row[2])  # doc_id
            u_d_items[0].append(row[3])  # isclick
            u_d_items[0].append(row[4])  # start_time
            u_d_items[0].append(row[5])  # session_id
            u_d_items[0].append(row[6])  # user_id
            u_d_items[0].append(row[7])  # doc_id
            # u_d_items[0].append(row[8])  # isclick
            # u_d_items[0].append(row[9])  # start_time
            # u_d_items[0].append(row[10])  # start_time
            session_dict[row[0]] = u_d_items
        else:
            u_d_item = []
            u_d_item.append(row[0])  # session_id
            u_d_item.append(row[1])  # user_id
            u_d_item.append(row[2])  # doc_id
            u_d_item.append(row[3])  # isclick
            u_d_item.append(row[4])  # start_time
            u_d_item.append(row[5])  # session_id
            u_d_item.append(row[6])  # user_id
            u_d_item.append(row[7])  # doc_id
            # u_d_item.append(row[8])  # isclick
            # u_d_item.append(row[9])  # start_time
            # u_d_item.append(row[10])  # start_time
            session_dict[row[0]].append(u_d_item)

    for key in session_dict.keys():
        u_d_pairs = session_dict[key]

        is_click = 0
        for i in range(len(u_d_pairs)):
            if u_d_pairs[i][3] == '1':
                is_click = 1
        if is_click == 1 and len(u_d_pairs) > 5 and random.random()<0.8:
            for i in range(len(u_d_pairs)):
                fp_out1.write(
                    u_d_pairs[i][0] + '\t' + u_d_pairs[i][1] + '\t' + u_d_pairs[i][2] + '\t' + u_d_pairs[i][
                        3] + '\t' +
                    u_d_pairs[i][4] + '\t' + u_d_pairs[i][5] + '\t' + u_d_pairs[i][6] + '\t' + u_d_pairs[i][7] )
                    # + '\t' + u_d_pairs[i][
                    #     8] + '\t' +
                    # u_d_pairs[i][9]+ '\t' +
                    # u_d_pairs[i][10])
        else:
            for i in range(len(u_d_pairs)):
                fp_out2.write(
                    u_d_pairs[i][0] + '\t' + u_d_pairs[i][1] + '\t' + u_d_pairs[i][2] + '\t' + u_d_pairs[i][
                        3] + '\t' +
                    u_d_pairs[i][4] + '\t' + u_d_pairs[i][5] + '\t' + u_d_pairs[i][6] + '\t' + u_d_pairs[i][7] )
                    # + '\t' + u_d_pairs[i][
                    #     8] + '\t' +
                    # u_d_pairs[i][9]+ '\t' +
                    # u_d_pairs[i][10])

#split_train_test_by_session_id_remove(ud_pairs_remove_dv_uv_file, ud_pairs_remove_dv_uv_session_file_train, ud_pairs_remove_dv_uv_session_file_test)
#split_train_val_test(udpairs_dv_uv_small, udpairs_dv_uv_train_val_raw_small, udpairs_dv_uv_test_raw_small)
#sample_train_val(udpairs_dv_uv_train_val_raw, udpairs_dv_uv_train_val_sample)
#split_train_val(udpairs_dv_uv_train_val_sample, udpairs_dv_uv_train, udpairs_dv_uv_val)

#remove_more_than_ten("./data/1--14-15-16/2018-11-01--2018-11-14_udPairs_click_doc_title_body_list.tsv", "./data/1--14-15-16/2018-11-01--2018-11-14_udPairs_click_doc_title_body_list_remove10.tsv")

#sample_11_15("./data/exp_compare/2018-11-15.standardized.udPairs.tsv", "./data/exp_compare/2018-11-15_sample.tsv")
#session_11_16("./data/exp_compare/2018-11-16.standardized.udPairs.tsv", "./data/exp_compare/2018-11-16_session.tsv")
#get_title_of_train_test("./data/exp_compare/2018-11-16_session_title.tsv", "./data/exp_compare/2018-11-16_session_title_only.tsv")
#combain_cdssm_for_train_test("D:/v-danyal/MSN/data/exp_compare/2018-11-16_session_title.tsv", "D:/v-danyal/MSN/data/exp_compare/2018-11-16_session_doc_cdssm.tsv", "D:/v-danyal/MSN/data/exp_compare/2018-11-16_session_all_cdssm.tsv")
#compute_avarage_user_cdssm("D:/v-danyal/MSN/data/1--14-15-16/2018-11-01--2018-11-14_udPairs_click_doc_title_body_list_cdssm.tsv")
#add_default_uservec_for_train_test("D:/v-danyal/MSN/data/exp_compare/2018-11-16_test.tsv", "D:/v-danyal/MSN/data/exp_compare/2018-11-16_test_f.tsv")
#deal_with_test("D:/v-danyal/MSN/data/exp_debug/2018-11-16_session_dv_uv.tsv","D:/v-danyal/MSN/data/exp_debug/2018-11-16_session_dv_uv_f.tsv")