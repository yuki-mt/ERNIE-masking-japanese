import unittest
from unittest.mock import Mock

from create_pretraining_data import create_masked_lm_predictions
from tokenization import CharToken, FullTokenizer
from tempfile import NamedTemporaryFile


class TrainingDataCreatorTest(unittest.TestCase):
  def setUp(self):
    with NamedTemporaryFile(mode='w') as tf:
      tf.write("a\n[CLS]\nb\n[SEP]c\nd\ne\nf\ng\nh\n")
      tf.seek(0)
      tokenizer = FullTokenizer(vocab_file=tf.name)
      self.vocab_words = list(tokenizer.vocab.keys())
    self.tokens = [CharToken('a', True), CharToken('b', False),
                   CharToken('c', False), CharToken('d', True),
                   CharToken('e', False), CharToken('f', True),
                   CharToken('g', False), CharToken('h', True)]

  def test_create_masked_lm_predictions(self):
    # always replace tokens with <MASK>
    random_mock = Mock()
    random_mock.random.return_value = 0.7

    output_tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
        tokens=self.tokens,
        masked_lm_prob=0.5,
        max_predictions_per_seq=5,
        vocab_words=self.vocab_words,
        rng=random_mock)
    self.assertEqual(masked_lm_positions, [0, 1, 2])
    self.assertEqual(masked_lm_labels, ['a', 'b', 'c'])
