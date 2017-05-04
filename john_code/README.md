This code has lots of dependency.

*R-related*

Please install R >= 3.3.3 and following R libraries:
- lme4: to compute random effect model 
- MuMIn: to compute r^2 of random effect model: http://onlinelibrary.wiley.com/doi/10.1111/j.2041-210x.2012.00261.x/full

Then, you have to install rpy2 to use r on Python3. I recommend Dock or virtual env but I'm not familiar with these.

*read data*
     fin_target = '/Users/forumai/Documents/work/stanford_work/all_item/large/morph/with_pos/choco/choco.model_outputs'
     fin_bp_in = '/Users/forumai/Documents/work/stanford_work/all_item/large/bpe/choco/choco.model_inputs.bpe'
     fin_morph_in = '/Users/forumai/Documents/work/stanford_work/all_item/large/morph/with_pos/choco/choco.model_inputs'

     test_item_id = '/Users/forumai/Documents/work/stanford_work/all_item/choco.multi_candid.all'

*BP encoding keywords*

   
     fin_bp_keyword = '/Users/forumai/Documents/work/stanford_work/GENERATED_WORDS/BPE/rnn_states-bahdanau-reverse_TRUE-   after_split-wv_size_16/choco-best-rnn_states-bahdanau-reverse_TRUE-after_split-wv_size_16'
    fin_odd_keyword = '/Users/forumai/Documents/work/stanford_work/all_item/all_odd_ratio/choco_morph.oddratio.txt'

    NUM_OF_TOP_KEYWORD = 500 #Number of keywords to use. recommend less than 1000. 

    MORPH_KEYWORD_LI = [line.strip().split()[0] for line in open(fin_odd_keyword) if float(line.strip().split()[1]) > 1.0][:NUM_OF_TOP_KEYWORD]
    BP_KEYWORD_LI = [line.strip().split()[0] for line in open(fin_bp_keyword) if len(line.strip().split()[0]) > 1][
                    :NUM_OF_TOP_KEYWORD]

*result*

Results give you abbliation test result for your neural model selection. this is to select best NN parameter for BPE/MORP. I think you can  use random_effect_r2 for the final selection. 

    =====result of keywords generated with mp=====
    result	fix_r2	random_effect_r2		adjusted
    all		0.4516	0.6860		0.5950
    -# of keyword	0.5936	0.6934		0.5950
    -pos	0.4470	0.7646		0.5943
    -bp	0.0321	0.6622		0.5107

features
- all =  use all features (pos + keyword+ BP(with NUM_OF_TOP_KEYWORD))
- wo/number of keyword = pos + BP(with NUM_OF_TOP_KEYWORD)
- wo/pos = keyword+ BP(with NUM_OF_TOP_KEYWORD)
- wo/BP = keyword+ pos

models
- fix_r2: this number is how well fixed effect variable(here, features above) can explain the sales.
- random_effect_r2: this number is how well fixed effect variable and random effect variable (shop_id, product_id,price) can explain sales
- adjusted^2: this number is the adjusted r^2 simple linear regression using both fixed and random effect (will be ignored)

*model selection*
- should be focused on all random effect_r2 (the second column of the first low)

