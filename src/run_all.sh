# ./run_all.sh ../data_wrangling/ 2

DATA=$1
GPU=$2

# run all combos
for wv_size in 16 32 64; do
    for reverse in True False; do
        for key in rnn_states word_vectors; do
            for type in bahdanau; do
                for order in before_split after_split; do
                    OUT_DIR=${key}-${type}-${reverse}-${order}-${wv-size}
                    python trainer.py ${OUT_DIR} \
                                      ${DATA}/total.inputs.bpe \
                                      ${DATA}/total.outputs \
                                      ${DATA}/bpe.vocab \
                                      --attention-type ${type} \
                                      --attention-keys ${key} \
                                      --attention-order ${order} \
                                      --gpu ${GPU} \
                                      --reverse-gradients ${reverse}
                                      --embedding-size ${wv_size}
                    python inference.py ${OUT_DIR} \
                                        ${DATA}/total.inputs.bpe \
                                        ${DATA}/total.outputs ${DATA}/bpe.vocab \
                                        --attention-type ${type} \
                                        --attention-keys ${key} \
                                        --attention-order ${order} \
                                        --gpu ${GPU} \
                                        --output ${OUT_DIR}/out.pkl 
                                        --reverse-gradients ${reverse}
                                        --embedding-size ${wv_size}
                    python pull_top_words.py ${key}-${type}-${reverse}/out.pkl \
                                             ${DATA}/health_multi_candid3.txt \
                                             ${DATA}/bpe.vocab \
                                             ${OUT_DIR}/health-best-${OUT_DIR} \
                                             ${OUT_DIR}/health-worst-${OUT_DIR}
                    python pull_top_words.py ${OUT_DIR}/out.pkl \
                                             ${DATA}/choco_multi_candid3.txt \
                                             ${DATA}/bpe.vocab \
                                             ${OUT_DIR}/choco-best-${OUT_DIR} \
                                             ${OUT_DIR}/choco-worst-${OUT_DIR}
                done
            done
        done
    done
done

# run baselines
python pull_random_words.py rnn_states-bahdanau-large/out.pkl \
                            ../data/labels/health_multi_candid3.txt \
                            ../data_wrangling/bpe.vocab \
                            health-random_baseline
python pull_random_words.py rnn_states-bahdanau-large/out.pkl \
                            ../data/labels/choco_multi_candid3.txt \
                            ../data_wrangling/bpe.vocab \
                            choco-random_baseline
