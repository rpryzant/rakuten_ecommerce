#  ./run_all.sh ../data/morph_small/ ../data/labels/ 1 MORPH > MORPH_OUT 2> MORPH_ERR
#  ./run_all.sh ../data/bpe_small/ ../data/labels/ 2 BPE > BPE_OUT 2> BPE_ERR

DATA=$1
TARGETS=$2
GPU=$3
OUT=$4

mkdir ${OUT}

# run all combos
for wv_size in 16 32 64; do
    for reverse in True False; do
        for key in word_vectors rnn_states; do
            for type in bahdanau; do
                for order in before_split after_split; do
                    SETTINGS=${key}-${type}-reverse_${reverse}-${order}-wv_size_${wv_size}
                    OUT_DIR=${OUT}/${SETTINGS}
                    echo 'INFO: starting '${OUT_DIR}
                    python trainer.py ${OUT_DIR} \
                                      ${DATA}/total.inputs \
                                      ${DATA}/total.outputs \
                                      ${DATA}/vocab \
                                      --attention-type ${type} \
                                      --attention-keys ${key} \
                                      --attention-order ${order} \
                                      --gpu ${GPU} \
                                      --reverse-gradients ${reverse} \
                                      --embedding-size ${wv_size} \
                                      --output ${OUT_DIR}/out.pkl 

                    # python inference.py ${OUT_DIR} \
                    #                     ${DATA}/total.inputs \
                    #                     ${DATA}/total.outputs \
                    #                     ${DATA}/vocab \
                    #                     --attention-type ${type} \
                    #                     --attention-keys ${key} \
                    #                     --attention-order ${order} \
                    #                     --gpu ${GPU} \
                    #                     --reverse-gradients ${reverse} \
                    #                     --embedding-size ${wv_size} \
                    #                     --output ${OUT_DIR}/out.pkl 
                    python pull_top_words.py ${OUT_DIR}/out.pkl \
                                             ${TARGETS}/health_multi_candid3.txt \
                                             ${DATA}/vocab \
                                             ${OUT_DIR}/health-best-${SETTINGS} \
                                             ${OUT_DIR}/health-worst-${SETTINGS}
                    python pull_top_words.py ${OUT_DIR}/out.pkl \
                                             ${TARGETS}/choco_multi_candid3.txt \
                                             ${DATA}/vocab \
                                             ${OUT_DIR}/choco-best-${SETTINGS} \
                                             ${OUT_DIR}/choco-worst-${SETTINGS}
                done
            done
        done
    done
done

# run baselines
python pull_random_words.py rnn_states-bahdanau-large/out.pkl \
                            ${TARGETS}/health_multi_candid3.txt \
                            ${DATA}/vocab \
                            ${OUT_DIR}/health-random_baseline
python pull_random_words.py rnn_states-bahdanau-large/out.pkl \
                            ${TARGETS}/choco_multi_candid3.txt \
                            ${DATA}/vocab \
                            ${OUT_DIR}/choco-random_baseline
