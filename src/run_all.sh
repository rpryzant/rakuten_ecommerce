# ./run_all.sh ../data_wrangling/ 2

DATA=$1
GPU=$2


for key in rnn_states word_vectors; do
    for type in bahdanau dot fc; do
        for size in large medium small; do
#            python trainer.py ${key}-${type}-${size} ${DATA}/total.inputs.bpe ${DATA}/total.outputs ${DATA}/bpe.vocab -t ${type} -k ${key} -s ${size} -g ${GPU}
            python inference.py ${key}-${type}-${size} ${DATA}/total.inputs.bpe ${DATA}/total.outputs ${DATA}/bpe.vocab -t ${type} -k ${key} -s ${size} -g ${GPU} -o ${key}-${type}-${size}/out.pkl 
            python pull_top_words.py ${key}-${type}-${size}/out.pkl ${DATA}/health_multi_candid3.txt ${DATA}/bpe.vocab ${key}-${type}-${size}/health_best ${key}-${type}-${size}/health_worst 
            python pull_top_words.py ${key}-${type}-${size}/out.pkl ${DATA}/choco_multi_candid3.txt ${DATA}/bpe.vocab ${key}-${type}-${size}/choco_best ${key}-${type}-${size}/choco_worst 
        done
    done
done


