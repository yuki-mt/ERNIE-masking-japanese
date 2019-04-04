import tensorflow as tf

from tokenization import SpecialToken, FullTokenizer, clean_text


def get_flags():
  flags = tf.flags
  flags.DEFINE_bool(
      "do_lower_case", True,
      "Whether to lower case the input text. Should be True for uncased "
      "models and False for cased models.")
  flags.DEFINE_string("mecab_userdic_path", None,
                      "path to mecab userdic (e.g. path to neologd)")
  flags.DEFINE_string("input_file", None,
                      "Input raw text file (or comma-separated list of files).")
  flags.DEFINE_string(
      "output_file", None,
      "Output TF example file (or comma-separated list of files).")
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  return flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = FullTokenizer(userdic_path=FLAGS.mecab_userdic_path,
                            do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  vocab = set()
  for input_file in input_files:
    tf.logging.info("Reading %s" % input_file)
    with tf.gfile.GFile(input_file, "r") as reader:
      for i, line in enumerate(reader):
        if i % 100000 == 0:
          tf.logging.info("done: %d" % i)
        line = clean_text(line.strip())
        if not line:
          continue
        tokens = tokenizer.tokenize(line)
        vocab |= set(t.value for t in tokens)
  tf.logging.info("*** Reading done ***")

  with open(FLAGS.output_file, "w") as f:
    for special_token in SpecialToken.all():
      f.write(special_token + "\n")
    for v in vocab:
      f.write(v + "\n")


if __name__ == "__main__":
  FLAGS = get_flags()
  tf.app.run()
