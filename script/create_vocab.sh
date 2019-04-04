python src/create_vocab.py \
  --input_file=./bert-japanese/data/wiki/*/all.txt \
  --output_file=data/char_ja_vocab.txt \
  --mecab_userdic_path=/usr/local/lib/mecab/dic/ipadic/mecab-user-dict-seed.dic
