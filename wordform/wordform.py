from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import codecs
import re

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils
from tensorflow.models.rnn.translate import seq2seq_model


tf.app.flags.DEFINE_float("learning_rate", 1.0, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("dev_set_size", 20000,
                            "Dev set size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 4000, "vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 5), (10, 10), (20, 20), (40, 40)]


STOP_WORD = re.compile("[\s|\d]+")

def char_tokenizer(sentence):
    words = []
    if "---" not in sentence:
        lw = ""
        for w in sentence.replace("\n","").decode("utf-8"):
            words.append(w.encode("utf-8"))
    return [w for w in words if w]

def bigram_tokenizer(sentence):
    words = []
    if "---" not in sentence:
        lw = ""
        for w in sentence.replace("\n","").decode("utf-8"):
            if lw:
                words.append((lw+w).encode("utf-8"))
                lw = ""
            else:
                lw = w
        if lw:
            words.append(lw.encode("utf-8"))
    return [w for w in words if w]

def load_pair_data(file_name):
    data = []
    with codecs.open(file_name,"r","latin-1") as f:
        for l in f:
            data.append(re.split("[\r|\n|\s|\|]+",l)[::2])
    return [d for d in data if len(d) == 2 and "/" not in d[0]]
#pair = load_pair_data("./data/autobild_1_02_2013_0_pair.txt")

def load_pair_data_from_dir(dir_path):
    pair = []
    for f in [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]:
        pair += load_pair_data(f)
    print("data loaded: ", len(pair))
    return pair

def save_data(dataset, file_name):
    with open(file_name) as f:
        for d in dataset:
            f.writeline(d)

def get_korpora_data_set(data_dir):
    valid_set_size = FLAGS.dev_set_size
    pair = load_pair_data_from_dir(os.path.join(data_dir,"korpora_tagged"))
    train_path = os.path.join(data_dir, "korpora_train")
    dev_path = os.path.join(data_dir, "korpora_dev")

    for (path, data) in zip([dev_path, train_path],[pair[:valid_set_size],pair[valid_set_size+1:]]):
        input_file = path +".input"
        target_file = path +".target"
        if not (tf.gfile.Exists(input_file) and tf.gfile.Exists(target_file)):
            print("creating new data file: %s, %s" % (input_file, target_file))
            with tf.gfile.GFile(input_file, mode="w") as ifile:
                with tf.gfile.GFile(target_file, mode="w") as tfile:
                    for d in data:
                        try:
                            ifile.write(d[0].encode("utf-8")+'\n')
                            tfile.write(d[1].encode("utf-8")+'\n')
                        except:
                            print(d)
                
    return train_path,dev_path

def prepare_korpora_data(data_dir, vocabulary_size):
  """Get WMT data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    en_vocabulary_size: size of the English vocabulary to create and use.
    fr_vocabulary_size: size of the French vocabulary to create and use.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for French training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for French development data-set,
      (5) path to the English vocabulary file,
      (6) path to the French vocabulary file.
  """
  # Get wmt data to the specified directory.
  train_path, dev_path = get_korpora_data_set(data_dir)
  #tokenizer = char_tokenizer
  tokenizer = bigram_tokenizer

  # Create vocabularies of the appropriate sizes.
  target_vocab_path = os.path.join(data_dir, "vocab%d.target" % vocabulary_size)
  input_vocab_path = os.path.join(data_dir, "vocab%d.input" % vocabulary_size)
  data_utils.create_vocabulary(target_vocab_path, train_path + ".target", vocabulary_size,tokenizer)
  data_utils.create_vocabulary(input_vocab_path, train_path + ".input", vocabulary_size,tokenizer)

  # Create token ids for the training data.
  target_train_ids_path = train_path + (".ids%d.target" % vocabulary_size)
  input_train_ids_path = train_path + (".ids%d.input" % vocabulary_size)
  data_utils.data_to_token_ids(train_path + ".target", target_train_ids_path, target_vocab_path,tokenizer)
  data_utils.data_to_token_ids(train_path + ".input", input_train_ids_path, input_vocab_path,tokenizer)

  # Create token ids for the development data.
  target_dev_ids_path = dev_path + (".ids%d.target" % vocabulary_size)
  input_dev_ids_path = dev_path + (".ids%d.input" % vocabulary_size)
  data_utils.data_to_token_ids(dev_path + ".target", target_dev_ids_path, target_vocab_path,tokenizer)
  data_utils.data_to_token_ids(dev_path + ".input", input_dev_ids_path, input_vocab_path,tokenizer)

  return (input_train_ids_path, target_train_ids_path,
          input_dev_ids_path, target_dev_ids_path,
          input_vocab_path, target_vocab_path)

def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.vocab_size,
      FLAGS.vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      use_lstm = True,
      forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

def ids2str(ids, rev_vocab):
    _REPLACED_START_VOCAB = [" ", "#", ".", "?"]
    string = ""
    for i in ids:
        if i < len(_REPLACED_START_VOCAB): 
            string += _REPLACED_START_VOCAB[i] 
        else:
            string += tf.compat.as_str(rev_vocab[i]) if i<len(rev_vocab) else " "
    return string

    
def evaluate_valid(model, session, dev_set, printed_size):
    # Load vocabularies.
    input_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.input" % FLAGS.vocab_size)
    target_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.target" % FLAGS.vocab_size)
    _, rev_input_vocab = data_utils.initialize_vocabulary(input_vocab_path)
    _, rev_target_vocab = data_utils.initialize_vocabulary(target_vocab_path)
    eval_datas = []
    for bucket_id in xrange(len(_buckets)):
          eval_datas_bucket = []
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, output_logits = model.step(session, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
              "inf")
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
          valid_input = np.transpose(encoder_inputs)
          valid_decode = np.transpose(decoder_inputs)
          for i in range(len(valid_input)):
            outputs = [int(np.argmax(logit[i:i+1], axis=1)) for logit in output_logits]
            
            #print(valid_input[i], valid_decode[i], outputs, [logit[i][int(np.argmax(logit[i:i+1], axis=1))] for logit in output_logits])
            
            if data_utils.EOS_ID in outputs:
              outputs = outputs[0:outputs.index(data_utils.EOS_ID)]
            istr, tstr, ostr = ids2str(valid_input[i][::-1], rev_input_vocab),ids2str(valid_decode[i], rev_target_vocab), ids2str(outputs, rev_target_vocab)
            eval_datas_bucket.append([istr, tstr, ostr])
            
          for i in range(min(printed_size, len(valid_input))):
            print("  sampled valid (i,t,o)",eval_datas_bucket[i][0], eval_datas_bucket[i][1], eval_datas_bucket[i][2])
          eval_datas += eval_datas_bucket  
          sys.stdout.flush()
    return eval_datas

def train():
  print("Preparing korpora data in %s" % FLAGS.data_dir)
  en_train, fr_train, en_dev, fr_dev, _, _ = prepare_korpora_data(
      FLAGS.data_dir, FLAGS.vocab_size)

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(en_dev, fr_dev)
    train_set = read_data(en_train, fr_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "korpora.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        
        # Run evals on development set and print their perplexity.
        input_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.input" % FLAGS.vocab_size)
        target_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.target" % FLAGS.vocab_size)
        _, rev_input_vocab = data_utils.initialize_vocabulary(input_vocab_path)
        _, rev_target_vocab = data_utils.initialize_vocabulary(target_vocab_path)
        for i in range(min(10, len(np.transpose(encoder_inputs)))):
            print("  sampled input (i,t,o)",
                ids2str(np.transpose(encoder_inputs)[i][::-1], rev_input_vocab),
                ids2str(np.transpose(decoder_inputs)[i], rev_target_vocab))
            
        evaluate_valid(model, sess, dev_set, 10)


def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.input" % FLAGS.vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.target" % FLAGS.vocab_size)
    en_vocab, rev_en_vocab = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab, bigram_tokenizer)
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      outputdata = [logit[0][int(np.argmax(logit, axis=1))] for logit in output_logits]
      print(token_ids,outputs, outputdata)
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print("".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)

def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()