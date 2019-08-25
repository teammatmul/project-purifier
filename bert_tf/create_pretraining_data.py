# coding=utf-8

# Copyright 2019 team Purifier
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License")


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tensorflow as tf
import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

# 변수설정 (dupe_factor -> 테스트 수 결정, vocab_file -> 한국어에 맞게 우리 vocab 사용)

flags.DEFINE_string("input_file", None, "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("output_file", None, "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text. Should be True for uncased models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20, "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer("dupe_factor", 10, "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float("short_seq_prob", 0.1, "Probability of creating sequences which are shorter than the maximum length.")


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()

# output_files 변수로 record 파일을 만드는 함수
def write_instance_to_example_files(instances, tokenizer, max_seq_length, max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []

  # 만들 파일 writers 에 append
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0
  total_written = 0


  for (inst_index, instance) in enumerate(instances):
    # instance.tokens을 ids로 변환 후 input_ids 변수에 저장
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    # input_mask 에 input_ids 에 만큼 1로 list 생성
    input_mask = [1] * len(input_ids)
    # instance.segment_ids를 리스트로 변경후 segment_ids 에 저장
    segment_ids = list(instance.segment_ids)

    # input_ids 가 max_seq_length 를 넘는지 체크 (넘으면 에러발생)
    assert len(input_ids) <= max_seq_length

    # input_ids 가 max_seq_length 보다 작을때까지 input_ids, input_mask, segment_ids 에 0 으로 append
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    # 제대로 0이 append 되었는지 체크
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    # masked_lm_positions 가 최대 max_predictions_per_seq 넘기 전까지 0 과 0.0을 append
    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    # is_random_next가 T면 1, F면 0으로 next_sentence_label 넣어줌
    next_sentence_label = 1 if instance.is_random_next else 0

    # features 의 dic 생성
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    # weights 변수만 float type 으로 생성
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    # tf_example에 features 를 넣어줌
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    #
    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    # total_written 수 증가
    total_written += 1

    # 20 feature 까지 example 로 출력해 보여줌
    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info("%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  # write 종료
  for writer in writers:
    writer.close()

  # 최종 total_written 수 출력
  tf.logging.info("Wrote %d total instances", total_written)

# train 을 int Feature 값으로 feature 생성
def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature

# train 을 Float Feature 값으로 feature 생성
def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


# input_files 에서 instances 데이터로 변경
def create_training_instances(input_files, tokenizer, max_seq_length, dupe_factor, short_seq_prob, masked_lm_prob, max_predictions_per_seq, rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # input file 에 대해서 다음 문장 예측이 있기 때문에 line 별로 읽어오지만 다음 라인은 앞 라인과 이어진 문장으로 생각
  # 그래서 우리의 task 에 맞게 input file 을 한문장 후 라인을 띄고 다음 문장을 입력 (아래에서 (2) 번 format으로 진행)

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.

  # input_files 을 읽어와 tokeniztion을 실행
  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
          all_documents[-1].append(tokens)

  # 빈 documents 삭제 후 all_documents 로 지정
  # Remove empty documents
  all_documents = [x for x in all_documents if x]

  # all_documents 섞기
  rng.shuffle(all_documents)

  # vocab keys 가 vocab_words
  vocab_words = list(tokenizer.vocab.keys())

  # instance를 담을 list 생성
  instances = []

  # dupe_factor 수 만큼 반복해서 documents 만큼 instance 생성
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)):
      instances.extend(create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob,
                                                      masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

  # instances 를 섞음
  rng.shuffle(instances)

  # instances 반환
  return instances


def create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob, masked_lm_prob,
                                   max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""

  document = all_documents[document_index]

  # max_seq_length 에서 특수 token 을 제외한 값을 max_num_tokens 로 설정
  # Account for [CLS], [SEP], [SEP] ([CLS] 는 문장의 시작, [SEP] 는 문장의 종료)
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.

  target_seq_length = max_num_tokens

  # 짧은 문자 비율? random 이 짧은 문장 비율로 나오면 target_seq_length 를 2와 max_num_tokens 사이의 값으로 바꿈
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long sequence and choose an arbitrary split point
  # because this would make the next sentence prediction task too easy.
  # Instead, we split the input into segments "A" and "B" based on the actual "sentences" provided by the user input.

  instances = []
  current_chunk = []
  current_length = 0

  i = 0

  while i < len(document):
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A` (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        # Random next
        is_random_next = False

        if len(current_chunk) == 1 or rng.random() < 0.5:
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large corpora.
          # However, just to be careful, we try to make sure that the random document is not the same as the document we're processing.
          for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break

          # We didn't actually use these segments so we "put them back" so they don't go to waste.
          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments

        # Actual next
        else:
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        # tokens, segment_ids 생성
        tokens = []
        segment_ids = []

        # tokens 에 [CLS] 시작 token append
        tokens.append("[CLS]")

        # segment_ids 에 0 append
        segment_ids.append(0)

        # tokens_a 의 token 을 tokens에, segment_ids에 0을 append 반복
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        # create_masked_Im_predictions() 함수사용
        (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)

        # instance 라는 TrainingInstance 클래스 생성
        instance = TrainingInstance(tokens=tokens, segment_ids=segment_ids, is_random_next=is_random_next, masked_lm_positions=masked_lm_positions, masked_lm_labels=masked_lm_labels)

        instances.append(instance)

      current_chunk = []
      current_length = 0
    i += 1

  return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


# token을 masking 하는 함수
def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    # 특수 token 일 경우 패스
    if token == "[CLS]" or token == "[SEP]":
      continue

    # cand_indexes list 에 index append
    cand_indexes.append(i)

  # cand_indexes list 를 셔플 시키는듯
  rng.shuffle(cand_indexes)

  # ouput_tokens 생성
  output_tokens = list(tokens)

  # num_to_predict 변수를 설정 (max_predict_per_seq 와 token에서 masking 비율을 곱한 후 1과 비교한 max 값에서 min 값으로 설정)
  num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []

  # 문장에서 masking 할 token의 index를 담을 set
  covered_indexes = set()

  for index in cand_indexes:
    # masked_lms 길이가 num_to_predict 길이 이상일때 break!
    if len(masked_lms) >= num_to_predict:
      break

    # index 번호가 coverd_indexes 에 이미 있을경우 패스
    if index in covered_indexes:
      continue

    # coverd_indexes set에 index를 add
    covered_indexes.add(index)

    masked_token = None

    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      # 10% of the time, keep original
      if rng.random() < 0.5:
        masked_token = tokens[index]

      # 10% 는 vocab dict 에서 랜덤하게 불러옴
      # 10% of the time, replace with random word
      else:
        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

    # output_tokens[index] 에 masked_token 을 넣어줌
    output_tokens[index] = masked_token

    #
    masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

  # masked_lms index 순으로 정렬
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []

  # masked_lm_positions 에 masked_lms index를 append / masked_lm_labels 에 label을 append.
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng)

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length, FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
