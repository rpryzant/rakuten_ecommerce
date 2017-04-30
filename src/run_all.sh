#  ./run_all.sh ../data/morph_small/ ../data/labels/ 1 MORPH > MORPH_OUT 2> MORPH_ERR
#  ./run_all.sh ../data/bpe_small/ ../data/labels/ 2 BPE > BPE_OUT 2> BPE_ERR

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
# ./run_all.sh ../data/large/morph/without_pos/health/ NA 0 MORPH_HEALTH > MORPH_HEALTH_OUT 2>MORPH_HEALTH_ERR
# ./run_all.sh ../data/large/morph/without_pos/choco/ NA 1 MORPH_CHOCO > MORPH_CHOCO_OUT 2>MORPH_CHOCO_ERR
# ./run_all.sh ../data/large/bpe/health NA 2 BPE_HEALTH > BPE_HEALTH_OUT 2>BPE_HEALTH_ERR
# ./run_all.sh ../data/large/bpe/choco NA 3 BPE_CHOCO> BPE_CHOCO_OUT 2>BPE_CHOCO_ERR



DATA=$1
TARGETS=$2
GPU=$3
OUT=$4

mkdir ${OUT}
echo ${GPU}
# run all combos
for wv_size in 64; do
    for reverse in True False; do
        for key in rnn_states word_vectors; do
            for type in bahdanau fc; do
                for order in before_split after_split; do
                    for mixing_ratio in 0.25 0.5 0.75; do
                        for hidden_size in 64; do
                            for attn_units in 64; do
                                for pred_units in 64; do
                                    SETTINGS=${wv_size}-${reverse}-${key}-${type}-${order}-${mixing_ratio}-${hidden_size}-${attn_units}-${pred_units}
                                    OUT_DIR=${OUT}/${SETTINGS}
				    PREDICTIONS=${OUT_DIR}/out.pkl
				    if [ -f $PREDICTIONS ]; then
					echo ${OUT_DIR}' already done!'
				    else
                                        echo 'INFO: starting '${OUT_DIR}
					python trainer.py ${OUT_DIR} \
                                                      ${DATA}/inputs \
                                                      ${DATA}/outputs \
                                                      ${DATA}/vocab \
                                                      --attention-type ${type} \
                                                      --attention-keys ${key} \
                                                      --attention-order ${order} \
                                                      --mixing-ratio ${mixing_ratio} \
                                                      --reverse-gradients ${reverse} \
                                                      --embedding-size ${wv_size} \
                                                      --hidden-size ${hidden_size} \
                                                      --attention-units ${attn_units} \
                                                      --prediction-units ${pred_units} \
                                                      --output ${OUT_DIR}/out.pkl \
                                                      --gpu ${GPU} \
                                    # python pull_top_words.py ${OUT_DIR}/out.pkl \
                                    #                          ${TARGETS}/health_multi_candid3.txt \
                                    #                          ${DATA}/vocab \
                                    #                          ${OUT_DIR}/health-best-${SETTINGS} \
                                    #                          ${OUT_DIR}/health-worst-${SETTINGS}
                                    # python pull_top_words.py ${OUT_DIR}/out.pkl \
                                    #                          ${TARGETS}/choco_multi_candid3.txt \
                                    #                          ${DATA}/vocab \
                                    #                          ${OUT_DIR}/choco-best-${SETTINGS} \
                                    #                          ${OUT_DIR}/choco-worst-${SETTINGS}
				    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# run baselines
# python pull_random_words.py rnn_states-bahdanau-large/out.pkl \
#                             ${TARGETS}/health_multi_candid3.txt \
#                             ${DATA}/vocab \
#                             ${OUT_DIR}/health-random_baseline
# python pull_random_words.py rnn_states-bahdanau-large/out.pkl \
#                             ${TARGETS}/choco_multi_candid3.txt \
#                             ${DATA}/vocab \
#                             ${OUT_DIR}/choco-random_baseline
