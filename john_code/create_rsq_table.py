import glob
from sklearn.preprocessing import StandardScaler
import math
import re
import numpy as np
import rpy2.robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
base = importr('base')
stats = importr('stats')
lme4 = importr('lme4')
MuMIn = importr('MuMIn')


data_li = []


POS_LI= ['動名詞','形容名詞','名詞形態指示詞','普通名詞','格助詞','名詞接頭辞','名詞性名詞接尾辞','カタカナ','句点','形容詞',
      '連体詞','副詞','判定詞','助動詞', '接続詞', '指示詞', '連体詞形態指示詞', '副詞形態指示詞', '感動詞', '名詞', '副詞的名詞',
      '形式名詞', '固有名詞', '組織名', '地名' ,'人名', 'サ変名詞', '数詞', '時相名詞', '動詞', '助詞','副助詞','接続助詞',
      '終助詞', '接頭辞', '動詞接頭辞', 'イ形容詞接頭辞', 'ナ形容詞接頭辞', '接尾辞', '名詞性述語接尾辞', '名詞性名詞助数辞',
      '名詞性特殊接尾辞', '形容詞性述語接尾辞', '形容詞性名詞接尾辞', '動詞性接尾辞', '特殊', '読点', '括弧始', '括弧終','記号','空白',
      '未定義語', 'アルファベット', 'その他', '複合名詞', '複合形容詞']


MORPH_KEYWORD_LI =[] #keyword selected by morph
BP_KEYWORD_LI = [] #keyword selected by BP

def create_item_keyword_dic(fin_target, fin_bp_in, fin_morph_in):

    """read bp and morph information"""
    item_bp_dic = {}
    #target_li read
    #feature_dic creation

    item_id_target_price_dic={}
    item_id_price_dic={}
    item_id_bpe_dic = {}
    item_id_morph_pos_dic = {}
    item_id_index_dic = {}
    item_id_pos_dic = {}

    item_count = 0

    for line in open(fin_target):

        line_items = line.strip().split('|')
        item_id = line_items[4]
        item_id_price_dic.setdefault(item_id,0)
        item_id_target_price_dic.setdefault(item_id,[0.0,0.0])
        item_id_index_dic[item_count]=item_id #to read bpe file
        item_count+=1

        item_price = 0

        try:
            item_price = int(line_items[2])
        except ValueError:
            #no info
            pass

        item_sales = float(line_items[0])
        if item_sales < 0.0:
            item_sales = -1
        else:
            pass

        item_id_target_price_dic[item_id][0]=item_sales
        item_id_target_price_dic[item_id][1] = item_price

    print(len(item_id_index_dic))

    line_count= 0
    ###start next processing -reading input file

    for line in open(fin_bp_in):
        line_items = line.strip().split()
        item_id =  item_id_index_dic[line_count]

        item_id_bpe_dic[item_id]=line_items
        line_count+=1

    print('reading bpe complete',len(item_id_bpe_dic))

    line_count=0
    for line in open(fin_morph_in):
        line_items = line.strip().split()
        item_id =  item_id_index_dic[line_count]
        item_id_morph_pos_dic.setdefault(item_id,[[],[]])
        item_id_pos_dic.setdefault(item_id,[])
        for line_item in line_items:
            try:
                word, pos = line_item.strip().split(':')
                if pos in POS_LI:

                    item_id_morph_pos_dic[item_id][0].append(word)
                    item_id_morph_pos_dic[item_id][1].append(pos)

            except ValueError:
                # :
                pass
        line_count+=1
    print('reading morph/pos complete',len(item_id_morph_pos_dic))

    #print(item_id_morph_pos_dic['kitanomori:10004209'])
    return(item_id_morph_pos_dic,item_id_bpe_dic, item_id_target_price_dic)



def read_desc(item_id_morph_pos_dic, item_id_bpe_dic, fin_test_item_id):

    """read description files (both morph and description file and read target item id for evaluation."""
    item_id_index_dic = {}
    index_item_id_dic = {}
    data_li = []
    item_count = 0

    print('# of pos in list:', len(POS_LI))
    print('# of bp keywords', len(BP_KEYWORD_LI))


    target_id_set = set()
    target_id_cate_dic = {}

    for line in open(fin_test_item_id):
        target_cate, target_id, _ = line.strip().split('\t')
        target_id_set.add(target_id)
        target_id_cate_dic[target_id] = target_cate

    print('# of items in test data', len(target_id_set))

    print(target_id_set)
    for item_id in item_id_morph_pos_dic.keys():
        if item_id not in target_id_set:
            # process item_id in the list only
            continue

        shop_id = item_id.split(':')[0]
        item_id_index_dic[item_id] = item_count
        index_item_id_dic[item_count] = item_id

        try:
            product_id = target_id_cate_dic[item_id]
        except KeyError:
            product_id = 'N'
            # for all data processing

        item_count += 1
        fea_dic = {}
        fea_dic['shop_id'] = shop_id
        fea_dic['product_id'] = product_id


        #create  bp keyword feature
        item_bp_list = item_id_bpe_dic[item_id]
        for bp_keyword in BP_KEYWORD_LI:
           if bp_keyword  in item_bp_list:

                bp_fea_name = 'bp.%s' % (bp_keyword)
                fea_dic.setdefault(bp_fea_name, 0)
                fea_dic[bp_fea_name] = 1

        # pos fea creation:
        tmp_pos_dic = {}



        morph_li, pos_li = item_id_morph_pos_dic[item_id][0], item_id_morph_pos_dic[item_id][1]

        if len(morph_li)!=len(pos_li):
            print('morph and pos have different length')

        for pos in pos_li:
            if pos in POS_LI:
                pos_fea_name = 'pos.%s' % (pos)
                tmp_pos_dic.setdefault(pos_fea_name, 0)
                tmp_pos_dic[pos_fea_name] += 1
                fea_dic.setdefault('keyword', 0)
                fea_dic['keyword'] += 1
            else:
                # print(pos)
                pass

        for morph in morph_li:
            if morph in MORPH_KEYWORD_LI:
                odd_fea_name = 'mp.%s' % (morph)
                fea_dic.setdefault(odd_fea_name, 0)
                fea_dic[odd_fea_name] = 1  # one-hot

        tmp_test = 0
        for pos_name, pos_count in tmp_pos_dic.items():
            # to makte ratio of pos

            fea_dic.setdefault(pos_name, 0)
            fea_dic[pos_name] = pos_count

            pos_ratio_name = pos_name + '.r'
            fea_dic.setdefault(pos_ratio_name, 0)
            fea_dic[pos_ratio_name] = pos_count / fea_dic['keyword']

            tmp_test += pos_count / fea_dic['keyword']

        data_li.append(fea_dic)

    print(data_li[0], index_item_id_dic[0])
    return (item_id_index_dic, index_item_id_dic, data_li)


def create_target_li(index_item_dic, data_li, item_id_target_price_dic) :
    target_li = [0] * len(data_li)

    for item_id, sales_info in item_id_target_price_dic.items():
        try:
            item_id_index = item_id_index_dic[item_id]
        except KeyError:

            # this item is not in evaluation set
            continue

        data_li[item_id_index]['price'] = sales_info[1]
        target_li[item_id_index] = sales_info[0]


    return (target_li)


def fea_vetorizer_manual(data_li):
    categorical_fea_set = {'shop_id', 'product_id', 'unit'}
    bp_fea_li = ['bp.%s' % (x) for x in BP_KEYWORD_LI]
    morph_fea_li = ['mp.%s' % (x) for x in MORPH_KEYWORD_LI]

    categorical_fea_set.update(set(bp_fea_li))
    categorical_fea_set.update(set(morph_fea_li))

    # print(len(categorical_fea_set))
    fea_li = ['keyword', 'price']

    fea_li.extend(['pos.%s' % (x) for x in POS_LI])
    fea_li.extend(['pos.%s.r' % (x) for x in POS_LI])

    converted_data_index = dict(zip(fea_li, range(0, len(fea_li))))
    # print(converted_data_index)
    converted_data_li = []

    tmp_li = [0.0] * len(fea_li)

    for item_fea_dic in data_li:
        # print(item_fea_dic)
        for k, v in item_fea_dic.items():
            if k in categorical_fea_set:
                pass
            else:
                tmp_li[converted_data_index[k]] = v
        converted_data_li.append(tmp_li)
        tmp_li = [0.0] * len(fea_li)

    scaled_data = StandardScaler().fit(converted_data_li).transform(converted_data_li)

    fea_li.extend(morph_fea_li)
    fea_li.extend(bp_fea_li)

    fea_li.append('shop_id')
    fea_li.append('product_id')
    print('# of total feature', len(fea_li))

    converted_data_li = []
    for i in range(0, len(scaled_data)):
        # deal with categorical values
        tmp_li = list(scaled_data[i])

        # odd keywords ratio
        tmp_morph_ratio_li = []
        for odd_key in morph_fea_li:

            try:
                tmp_morph_ratio_li.append(data_li[i][odd_key])
            except KeyError:
                tmp_morph_ratio_li.append(0)

        tmp_li.extend(tmp_morph_ratio_li)

        tmp_bp_ratio_li = []
        for bp_key in bp_fea_li:
            try:
                tmp_bp_ratio_li.append(data_li[i][bp_key])
            except KeyError:
                tmp_bp_ratio_li.append(0)

        tmp_li.extend(tmp_bp_ratio_li)

        shop_id = data_li[i]['shop_id']
        cate = data_li[i]['product_id']
        tmp_li.append(shop_id)
        tmp_li.append(cate)

        converted_data_li.append(tmp_li)

    # print(len(tmp_li))
    print(converted_data_li[0])
    return (converted_data_li, fea_li)

def convert_to_rdata(scaled_data, fea_li, target_li):
    test_count = len(target_li)
    # conver data for R
    r_input = np.concatenate((np.array(scaled_data), np.array(target_li).reshape(test_count, 1)), axis=1)
    r_feature = fea_li[:]
    r_feature.append('target')
    r_input_t = r_input.T
    r_df = {}
    for i in range(0, len(r_feature)):
        # print(r_feature[i])
        fea_name = r_feature[i]
        if fea_name == 'shop_id' or fea_name == 'product_id':
            r_df[fea_name] = rpy2.robjects.FactorVector(r_input_t[i])

        else:
            r_df[fea_name] = rpy2.robjects.vectors.FloatVector(r_input_t[i])

    dataf = rpy2.robjects.DataFrame(r_df)

    return(dataf)


if __name__ == '__main__':
    fin_target =  '/Users/forumai/Documents/work/stanford_work/all_item/large/morph/with_pos/choco/choco.model_outputs'
    fin_bp_in = '/Users/forumai/Documents/work/stanford_work/all_item/large/bpe/choco/choco.model_inputs.bpe'
    fin_morph_in ='/Users/forumai/Documents/work/stanford_work/all_item/large/morph/with_pos/choco/choco.model_inputs'


    test_item_id = '/Users/forumai/Documents/work/stanford_work/item_id_desc/choco_multi_candid3.txt'

    fin_bp_keyword =  '/Users/forumai/Documents/work/stanford_work/GENERATED_WORDS/BPE/rnn_states-bahdanau-reverse_TRUE-after_split-wv_size_16/choco-best-rnn_states-bahdanau-reverse_TRUE-after_split-wv_size_16'
    fin_odd_keyword = '/Users/forumai/Documents/work/stanford_work/item_id_desc/sales_regression/choco.odd_ratio.wordcate.txt'

    MORPH_KEYWORD_LI = [line.strip().split()[0] for line in open(fin_odd_keyword) if
                        float(line.strip().split()[1]) > 1.0]
    NUM_OF_TOP_KEYWORD = 358
    BP_KEYWORD_LI = [line.strip().split()[0] for line in open(fin_bp_keyword) if len(line.strip().split()[0]) > 1][:NUM_OF_TOP_KEYWORD]

    item_id_morph_pos_dic, item_id_bpe_dic, item_id_target_price_dic = create_item_keyword_dic(fin_target, fin_bp_in, fin_morph_in)
    item_id_index_dic, index_item_id_dic, data_li = read_desc(item_id_morph_pos_dic, item_id_bpe_dic, test_item_id)
    target_li = create_target_li(index_item_id_dic, data_li,item_id_target_price_dic)

    scaled_data, feature_name = fea_vetorizer_manual(data_li) # data for scikit

    #print(feature_name)

    #start R code
    dataf = convert_to_rdata(scaled_data, feature_name, target_li)

    rpy2.robjects.globalenv['dataset'] = dataf #append dataframe to environment
    pos_index = rpy2.robjects.r('''grep("^pos", colnames(dataset))''')
    bp_index = rpy2.robjects.r('''grep("^bp", colnames(dataset))''')
    mp_index = rpy2.robjects.r('''grep("^mp", colnames(dataset))''')

    posbp_index  = np.append(pos_index,bp_index)
    #print(pos_index,bp_index,posbp_index)

    posmp_index  = np.append(pos_index, mp_index)

    mpbp_index = np.append(mp_index, bp_index)


    bp_index_neg = rpy2.robjects.IntVector(tuple([x*-1 for x in bp_index]))
    mp_index_neg = rpy2.robjects.IntVector(tuple([x * -1 for x in mp_index]))
    posbp_index_neg = rpy2.robjects.IntVector(tuple([x * -1 for x in posbp_index]))
    posmp_index_neg = rpy2.robjects.IntVector(tuple([x * -1 for x in posmp_index]))
    mpbp_index_neg =  rpy2.robjects.IntVector(tuple([x * -1 for x in mpbp_index]))

    # selection from df

    rpy2.robjects.globalenv['dataset_wo_bp'] = dataf.rx(True, bp_index_neg)
    rpy2.robjects.globalenv['dataset_wo_mp'] = dataf.rx(True, mp_index_neg)
    rpy2.robjects.globalenv['dataset_wo_posbp'] = dataf.rx(True, posbp_index_neg)
    rpy2.robjects.globalenv['dataset_wo_posmp'] = dataf.rx(True, posmp_index_neg)
    rpy2.robjects.globalenv['dataset_wo_mpbp'] = dataf.rx(True, mpbp_index_neg)


    #contain all features w/o product_id. i.e. containing shop and price for usual regression
    #this is because regular regression R^2 is to see whether all features can explain sales well


 
    result = rpy2.robjects.r('''fit=lmer(target ~ . -price  -shop_id -product_id  + (1|shop_id) + (1|product_id), data=dataset_wo_mp)''')
    rpy2.robjects.globalenv['lm_result'] = result
    all_result = list(rpy2.robjects.r('''r.squaredGLMM(lm_result)'''))#language only vs # language/shop/product_id

    result = rpy2.robjects.r('''fit=lm(target ~ . -product_id, data=dataset_wo_mp)''')
    all_result.append(float(base.summary(result)[8][0]))

    #print(base.summary(result))

    #all - #of keyword
    result = rpy2.robjects.r('''fit=lmer(target ~ . -keyword -price  -shop_id -product_id  + (1|shop_id) + (1|product_id), data=dataset_wo_mp)''')
    rpy2.robjects.globalenv['lm_result'] = result
    result_wo_keyword = list(rpy2.robjects.r('''r.squaredGLMM(lm_result)''')) #language only vs # language/shop/product_id


    result = rpy2.robjects.r('''fit=lm(target ~ . -keyword -product_id, data=dataset_wo_mp)''')
    result_wo_keyword.append(float(base.summary(result)[8][0]))  # adjusted_r



    #all - pos
    result = rpy2.robjects.r('''fit=lmer(target ~ . -price  -shop_id -product_id  + (1|shop_id) + (1|product_id), data=dataset_wo_posmp)''')
    rpy2.robjects.globalenv['lm_result'] = result
    result_wo_pos = list(rpy2.robjects.r('''r.squaredGLMM(lm_result)''')) #language only vs # language/shop/product_id

    result = rpy2.robjects.r('''fit=lm(target ~ . -product_id, data=dataset_wo_posmp)''')
    result_wo_pos.append(float(base.summary(result)[8][0]))  # adjusted_r



    #all - bp

    result = rpy2.robjects.r(
        '''fit=lmer(target ~ . -price  -shop_id -product_id  + (1|shop_id) + (1|product_id), data=dataset_wo_mpbp)''')
    rpy2.robjects.globalenv['lm_result'] = result
    result_wo_bp = list(rpy2.robjects.r('''r.squaredGLMM(lm_result)'''))  # language only vs # language/shop/product_id

    result = rpy2.robjects.r('''fit=lm(target ~ . -product_id, data=dataset_wo_mpbp)''')
    result_wo_bp.append(float(base.summary(result)[8][0]))  # adjusted_r

    print('=====result of keywords generated with bp=====')
    print('result\tfix_r2\trandom_effect_r2\t\tadjusted')
    print('all\t\t%.4f\t%.4f\t\t%.4f' % (all_result[0], all_result[1], all_result[2]))
    print('-# of keyword\t%.4f\t%.4f\t\t%.4f' % (result_wo_keyword[0], result_wo_keyword[1], result_wo_keyword[2]))
    print('-pos\t%.4f\t%.4f\t\t%.4f' % (result_wo_pos[0], result_wo_pos[1], result_wo_pos[2]))
    print('-bp\t%.4f\t%.4f\t\t%.4f' % (result_wo_bp[0], result_wo_bp[1], result_wo_bp[2]))



    result = rpy2.robjects.r(
        '''fit=lmer(target ~ . -price -shop_id -product_id  + (1|shop_id) + (1|product_id), data=dataset_wo_bp)''')
    rpy2.robjects.globalenv['lm_result'] = result
    all_result = list(rpy2.robjects.r('''r.squaredGLMM(lm_result)'''))  # language only vs # language/shop/product_id

    result = rpy2.robjects.r('''fit=lm(target ~ . -product_id, data=dataset_wo_bp)''')
    all_result.append(float(base.summary(result)[8][0]))

    # print(base.summary(result))



    # all - #of keyword
    result = rpy2.robjects.r(
        '''fit=lmer(target ~ . -keyword -price + -shop_id -product_id  + (1|shop_id) + (1|product_id), data=dataset_wo_bp)''')
    rpy2.robjects.globalenv['lm_result'] = result
    result_wo_keyword = list(rpy2.robjects.r('''r.squaredGLMM(lm_result)'''))  # language only vs # language/shop/product_id

    result = rpy2.robjects.r('''fit=lm(target ~ . -keyword -product_id, data=dataset_wo_bp)''')
    result_wo_keyword.append(float(base.summary(result)[8][0]))  # adjusted_r

    # all - pos
    result = rpy2.robjects.r(
        '''fit=lmer(target ~ . -price + -shop_id -product_id  + (1|shop_id) + (1|product_id), data=dataset_wo_posbp)''')
    rpy2.robjects.globalenv['lm_result'] = result
    result_wo_pos = list(rpy2.robjects.r('''r.squaredGLMM(lm_result)'''))  # language only vs # language/shop/product_id

    result = rpy2.robjects.r('''fit=lm(target ~ . -product_id, data=dataset_wo_posbp)''')
    result_wo_pos.append(float(base.summary(result)[8][0]))  # adjusted_r

    # all - mp

    result = rpy2.robjects.r(
        '''fit=lmer(target ~ . -price + -shop_id -product_id  + (1|shop_id) + (1|product_id), data=dataset_wo_mpbp)''')
    rpy2.robjects.globalenv['lm_result'] = result
    result_wo_mp = list(rpy2.robjects.r('''r.squaredGLMM(lm_result)'''))  # language only vs # language/shop/product_id

    result = rpy2.robjects.r('''fit=lm(target ~ . -product_id, data=dataset_wo_mpbp)''')
    result_wo_mp.append(float(base.summary(result)[8][0]))  # adjusted_r


    print('=====result of keywords generated with mp=====')
    print('result\tfix_r2\trandom_effect_r2\t\tadjusted')
    print('all\t\t%.4f\t%.4f\t\t%.4f' % (all_result[0], all_result[1], all_result[2]))
    print('-# of keyword\t%.4f\t%.4f\t\t%.4f' % (result_wo_keyword[0], result_wo_keyword[1], result_wo_keyword[2]))
    print('-pos\t%.4f\t%.4f\t\t%.4f' % (result_wo_pos[0], result_wo_pos[1], result_wo_pos[2]))
    print('-bp\t%.4f\t%.4f\t\t%.4f' % (result_wo_bp[0], result_wo_mp[1], result_wo_mp[2]))

