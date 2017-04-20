# ./run_all.sh ../data_wrangling/ word_vectors 2

DATA=$1
GPU=$2


for key in rnn_states word_vectors; do
    for type in bahdanau dot fc; do
        for size in large medium small; do
            python trainer.py ${key}-${type}-${size} ${DATA}/total.inputs.bpe ${DATA}/total.outputs ${DATA}/bpe.vocab -t ${type} -k ${key} -s ${size} -g ${GPU}
        done
    done
done
