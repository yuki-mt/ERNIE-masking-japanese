from unittest import TestCase
import os
from tokenization import FullTokenizer, CharToken


class TokenizationTest(TestCase):
  def test_tokenize(self):
    tokenizer = FullTokenizer()
    sentence = '実質的変化はなかった'
    res = tokenizer.tokenize(sentence)
    firsts = [0, 2, 3, 5, 6, 9]
    tokens = [CharToken(c, is_first=i in firsts) for i, c in enumerate(sentence)]
    self.assertEqual(res, tokens)

  def test_tokenize_with_nelogd(self):
    NEOLOGD_PATH = "/usr/local/lib/mecab/dic/ipadic/mecab-user-dict-seed.dic"
    if not os.path.isfile(NEOLOGD_PATH):
      raise ValueError('NEOLOGD_PATH is invalid. Please set a file path to neologd dic')
    sentence = '実質的変化はなかった'
    tokenizer = FullTokenizer(userdic_path=NEOLOGD_PATH)
    firsts = [0, 3, 5, 6, 9]
    tokens = [CharToken(c, is_first=i in firsts) for i, c in enumerate(sentence)]
    res = tokenizer.tokenize(sentence)
    self.assertEqual(res, tokens)
