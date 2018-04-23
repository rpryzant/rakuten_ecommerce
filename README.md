_This repository is **DEPRECIATED**. 
Please see [this repo](https://github.com/rpryzant/deconfounded_lexicon_induction) for an official code release._


# Rakuten Ecommerce
Discovering the best parts of ecommerce product descriptions. Data courtesy of [Rakuten](http://global.rakuten.com/en/). 


## Data Description

* **Description files**
  * Filename: `<type>.desc.<ID>`
  * Format: `<item title> <item_id> <price> <description> <item_id(not used)><img_url><number of review> <avg_rating> <shop_name> <category_id>`

* **Sales files**
   * Filename: `<type>_sales.txt`
   * Format: `<item_id> <sold unit>`
   
* **Note**
   * Files are tab-seperated
   
## Approach

Use adversarial discriminative network to find subsequences of `description` that are predictive of `log(sold_unit)`, irrespective of other factors (title, price, shop rating, category id).
