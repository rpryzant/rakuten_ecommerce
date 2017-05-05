This code has lots of dependency.

*R-related*

Please install R >= 3.3.3 and following R libraries:
- lme4: to compute random effect model 
- MuMIn: to compute r^2 of random effect model: http://onlinelibrary.wiley.com/doi/10.1111/j.2041-210x.2012.00261.x/full

Then, you have to install rpy2 to use r on Python3. I recommend Dock or virtual env but I'm not familiar with these.

*read data*
     fin_target = '/Users/forumai/Documents/work/stanford_work/all_item/large/morph/with_pos/choco/choco.model_outputs'
     fin_bp_in = '/Users/forumai/Documents/work/stanford_work/all_item/large/bpe/choco/choco.model_inputs.bpe'
     fin_morph_keyword = '/Users/forumai/Documents/work/stanford_work/GENERATED_WORDS/MORPH/rnn_states-bahdanau-reverse_True-after_split-wv_size_16/choco-best-rnn_states-bahdanau-reverse_TRUE-after_split-wv_size_16'


     test_item_id = '/Users/forumai/Documents/work/stanford_work/all_item/choco.multi_candid.all'

*BP encoding keywords*

   
     fin_bp_keyword = '/Users/forumai/Documents/work/stanford_work/GENERATED_WORDS/BPE/rnn_states-bahdanau-reverse_TRUE-   after_split-wv_size_16/choco-best-rnn_states-bahdanau-reverse_TRUE-after_split-wv_size_16'

    NUM_OF_TOP_KEYWORD = 500 #Number of keywords to use. recommend less than 1000. 

    MORPH_KEYWORD_LI = [line.strip().split()[0] for line in codecs.open(fin_morph_keyword, 'r', encoding='utf-8') if len(line.strip().split()[0]) > 1][:NUM_OF_TOP_KEYWORD]
    BP_KEYWORD_LI = [line.strip().split()[0] for line in open(fin_bp_keyword) if len(line.strip().split()[0]) > 1][
                    :NUM_OF_TOP_KEYWORD]

*result*

Results give you abbliation test result for your neural model selection. this is to select best NN parameter for BPE/MORP. I think you can  use random_effect_r2 for the final selection. 

This result is show that how shop_id and product_id's effect can explain the sales.


    =====result of random effect only (shop id / product_id)=====
    result	fix_r2	random_effect_r2		adjusted
    all		0.0000	0.5177		0.5884    

These results are for the table for BPE/MORH.
  
    =====result of keywords generated with bp=====
    result	fix_r2	random_effect_r2		adjusted
    all		0.4539	0.7469		0.7014
    BP + POS	0.4539	0.7469		0.7014
    BP + # keywords 	0.4404	0.7381		0.6861
    BP only	0.4319	0.7386		0.5120

    
    =====result of keywords generated with mp=====
    result	fix_r2	random_effect_r2		adjusted
    all		0.3623	0.7434		0.6444
    MORPH + POS	0.3623	0.7434		0.6444
    MORPH + # keyword 	0.3461	0.7461		0.6286
    MORPH only	0.3465	0.7462		0.6293

features
- all =  use all features (pos + keyword+ BP(with NUM_OF_TOP_KEYWORD))

models
- fix_r2: this number is how well fixed effect variable(here, features above) can explain the sales.
- random_effect_r2: this number is how well fixed effect variable and random effect variable (shop_id, product_id,price) can explain sales
- adjusted^2: this number is the adjusted r^2 simple linear regression using both fixed and random effect (will be ignored)

*model selection*
- should be focused on all random effect_r2 (the second column of the first low)

