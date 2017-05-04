import sys

def compute_odd_ratio(fin_in, fin_out, fout):
    item_id_line_dic = {}


    line_label_dic = {}
    line_count = 0
    for line in open(fin_out):
        line_items = line.strip().split('|')
        line_label_dic[line_count]=line_items[0]
        line_count+=1

    print(line_label_dic)

    all_high_item = 0
    all_low_item = 0


    line_count= 0


    keyword_high_low_dic = {}
    for line in open(fin_in):
        line_items = line.strip().split()

        line_label = line_label_dic[line_count]

        for keyword in line_items:
            keyword_high_low_dic.setdefault(keyword,[0,0])

            if line_label == '1':
                keyword_high_low_dic[keyword][0] += 1
                all_high_item+=1
            else:
                keyword_high_low_dic[keyword][1] += 1
                all_low_item+=1
        line_count+=1

    print(len(keyword_high_low_dic))


    odd_ratio_dic = {}
    rare_word_lines = []
    for word in keyword_high_low_dic.keys():
        odd_ratio_dic.setdefault(word, 0.0)

        a = keyword_high_low_dic[word][0]  # high_exist
        b = keyword_high_low_dic[word][1]  # low_exist
        c = all_high_item - a #high_non_exist
        d = all_low_item - b #low_non_exist

        try:
            #high
            odd_ratio_dic[word] = (a * d) / (b * c)

            # low
        except ZeroDivisionError:
            print(word, keyword_high_low_dic[word])
            rare_word_lines.append('%s\t\%s\n'%(word, '\t'.join(map(str, keyword_high_low_dic[word]))))

    sorted_odd_ratio = sorted(odd_ratio_dic.items(), key=lambda x: x[1], reverse=True)
    open(fout, 'w').writelines(['%s\t%.3f\n' % (x[0], x[1]) for x in sorted_odd_ratio])



if __name__=='__main__':
    fin_in = '/Users/forumai/Documents/work/stanford_work/all_item/all_odd_ratio/morph/without_pos/health/inputs.binary'
    fin_out = '/Users/forumai/Documents/work/stanford_work/all_item/all_odd_ratio/morph/without_pos/health/outputs.binary'
    fout = '/Users/forumai/Documents/work/stanford_work/all_item/all_odd_ratio/health_morph.oddratio.txt'

    compute_odd_ratio(fin_in, fin_out, fout)
