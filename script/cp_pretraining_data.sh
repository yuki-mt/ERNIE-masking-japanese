for DIR in $( find /work/data/wiki/ -mindepth 1 -type d ); do
  OUTPUT_FILE=${DIR}/all-maxseq128.tfrecord
  NEW_DIR="/work/data/wiki_record/$(basename $DIR)"
  mkdir -p $NEW_DIR
  cp $OUTPUT_FILE $NEW_DIR
done
