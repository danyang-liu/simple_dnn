from conf import *
import csv

def analysis_doc_feature(file):
    line_num = 0

    doc_num = 0
    doc_dict = {}

    type = {}

    vert = {}
    subvert = {}
    with open(file, encoding="utf8") as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            line_num = line_num + 1
            if row[0] not in doc_dict:
                doc_dict[row[0]] = 1
                doc_num = doc_num + 1
            else:
                doc_dict[row[0]] = doc_dict[row[0]] + 1

            if row[1] not in type:
                type[row[1]] = 1
            else:
                type[row[1]] = type[row[1]] + 1

            if row[2] not in vert:
                vert[row[2]] = 1
            else:
                vert[row[2]] = vert[row[2]] + 1

            if row[3] not in subvert:
                subvert[row[3]] = 1
            else:
                subvert[row[3]] = subvert[row[3]] + 1
    pass

def analysis_user_click_history(file):
    user_num =0
    user_dict= {}
    line_num = 0
    id_type = {}
    ip_state = {}
    doc_dict = {}
    doc_num = 0
    with open(file, encoding="utf8") as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            line_num = line_num + 1

            if row[0] not in user_dict:
                user_num = user_num + 1
                user_dict[row[0]] = 1
            else:
                user_dict[row[0]] = user_dict[row[0]] + 1

            if row[1] not in id_type:
                id_type[row[1]] = 1
            else:
                id_type[row[1]] = id_type[row[1]] + 1

            if row[2] not in ip_state:
                ip_state[row[2]] = 1
            else:
                ip_state[row[2]] = ip_state[row[2]] + 1

            if row[5] not in doc_dict:
                doc_num = doc_num + 1
                doc_dict[row[5]] = 1
            else:
                doc_dict[row[5]] = doc_dict[row[5]] + 1

    pass

def anaylsis_ud_paris(file):
    page_view_count = [0,0,0,0,0,0,0]
    id_type = {}
    user_num = 0
    user_dict = {}
    line_num = 0
    doc_dict = {}
    doc_num = 0
    with open(file, encoding="utf8") as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            line_num = line_num + 1
            if row[4] not in doc_dict:
                doc_num = doc_num + 1
                doc_dict[row[4]] = 1
            else:
                doc_dict[row[4]] = doc_dict[row[4]] + 1

            if row[1] not in user_dict:
                user_num = user_num + 1
                user_dict[row[1]] = int(row[3])
            # else:
            #     user_dict[row[1]] = user_dict[row[1]] + 1

            if row[2] not in id_type:
                id_type[row[2]] = 1
            else:
                id_type[row[2]] = id_type[row[2]] + 1

    for key in user_dict:
        if int(user_dict[key]) <= 10:
            page_view_count[0] = page_view_count[0] + 1
        elif int(user_dict[key]) <= 100:
            page_view_count[1] = page_view_count[1] + 1
        elif int(user_dict[key]) <= 500:
            page_view_count[2] = page_view_count[2] + 1
        elif int(user_dict[key]) <= 1000:
            page_view_count[3] = page_view_count[3] + 1
        elif int(user_dict[key]) <= 2000:
            page_view_count[4] = page_view_count[4] + 1
        elif int(user_dict[key]) <= 5000:
            page_view_count[5] = page_view_count[5] + 1
        else:
            page_view_count[6] = page_view_count[6] + 1



    pass

#analysis_user_click_history(user_click_history_file)
# analysis_doc_feature(doc_features_file)
# anaylsis_ud_paris(ud_pairs_remove_dv_uv_file)