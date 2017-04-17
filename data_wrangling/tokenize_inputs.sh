
CORPUS=$1

# learn BPE
spm_train \
  --input=${CORPUS} \
  --model_prefix=bpe \
  --vocab_size=32000 \
  --model_type=bpe

# apply BPE
spm_encode --model=bpe.model --output_format=piece \
  < ${CORPUS} \
  > ${CORPUS1}.bpe
