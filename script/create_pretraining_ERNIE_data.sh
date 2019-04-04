for DIR in $( find /work/data/wiki/ -mindepth 1 -type d ); do
  NEW_DIR="/work/data/wiki_record_ERNIE/$(basename $DIR)"
  mkdir -p $NEW_DIR
  OUTPUT_FILE=${NEW_DIR}/all-maxseq128.tfrecord
  if [ -e $OUTPUT_FILE ]; then
    echo "skip $DIR because output file already exists"
    continue
  fi
  nohup python3 src/create_pretraining_data.py \
    --input_file=${DIR}/all.txt \
    --output_file=${OUTPUT_FILE} \
    --vocab_file=./data/char_ja_vocab.txt \
    --do_lower_case=True \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=5 &
done
