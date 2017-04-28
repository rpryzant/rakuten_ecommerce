This code has lots of dependency.  (But at least it's working haha)

*R-related*

You have to install R >= 3.3.3 and you have to install following R libraries:
- lme4: to compute random effect model 
- MuMIn: to compute r^2 of random effect model: http://onlinelibrary.wiley.com/doi/10.1111/j.2041-210x.2012.00261.x/full

Then, you have to install rpy2 to use r on Python3. I recommend Dock or virtual env but I'm not familiar with these.

*read data*
   
    fin_desc_pro_pattern = '/Users/forumai/Documents/work/stanford_work/item_id_desc/choco_desc_pro/choco.desc.*.pre'
    fin_desc_pattern = '/Users/forumai/Documents/work/stanford_work/item_id_desc/choco_desc/choco.desc.*[0-9]'

    fin_desc_head = '/Users/forumai/Documents/work/stanford_work/item_id_desc/choco_desc/choco.desc.'
    fin_sales = '/Users/forumai/Documents/work/stanford_work/item_id_desc/sales_regression/choco_all.sales2.combined.txt'

These files will be replaced with the files you're processing. So need to clean up the read_data and read_sales

*BP encoding keywords*

    fin_bp_keyword =  '/Users/forumai/Documents/work/stanford_work/GENERATED_WORDS/BPE/rnn_states-bahdanau-reverse_TRUE-after_split-wv_size_16/choco-best-rnn_states-bahdanau-reverse_TRUE-after_split-wv_size_16' #file path
    NUM_OF_TOP_KEYWORD = 300 # of keyword to use
    BP_KEYWORD_LI = [line.strip().split()[0] for line in open(fin_bp_keyword) if len(line.strip().split()[0]) > 1][:NUM_OF_TOP_KEYWORD]

*result*

Results give you abbliation test result for your neural model selection. this is to select best NN parameter for BPE. I think you can  use random_effect_r2 for the final selection. 

    result	fix_r2	random_effect_r2		adjusted
    all		0.7309   0.8114		0.7654
    -# of keyword	0.7309	0.8114		0.7654
    -pos	0.5550	0.8127		0.6317
    -bp	0.5106	0.7878		0.6186

features
- all =  use all features (pos + keyword+ BP(with NUM_OF_TOP_KEYWORD))
- wo/number of keyword = pos + BP(with NUM_OF_TOP_KEYWORD)
- wo/pos = keyword+ BP(with NUM_OF_TOP_KEYWORD)
- wo/BP = keyword+ pos

models
- fix_r2: this number is how well fixed effect variable(here, features above) can explain the sales.
- random_effect_r2: this number is how well fixed effect variable and random effect variable (shop_id, product_id,price) can explain sales
- adjusted^2: this number is the adjusted r^2 simple linear regression using both fixed and random effect.
