
CORPUS=$1

# learn BPE
spm_train \
  --input=${CORPUS} \
  --model_prefix=bpe \
  --vocab_size=20000 \
  --model_type=bpe

# apply BPE
spm_encode --model=bpe.model --output_format=piece \
  < ${CORPUS} \
  > ${CORPUS}.bpe
