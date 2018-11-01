#! /usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
import re
import os
import time

# Parameters
tf.flags.DEFINE_string("data_file", "./data/data.txt", "Input Data File Path")
tf.flags.DEFINE_string("class_file", "./data/class.txt", "Output Class File Path")

FLAGS = tf.flags.FLAGS

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    # liyc: Delete Continuous Space
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def code_to_class(string):
    string = re.sub(r"\s", "", string)
    return string.strip()

def load_data_labels(data_file, class_file):
    data_examples = list(open(data_file, 'r', encoding="utf-8").readlines())
    data_examples = [s.strip() for s in data_examples]

    class_examples = list(open(class_file, 'r', encoding="utf-8").readlines())
    class_examples = [code_to_class(s) for s in class_examples]

    #data_text = [clean_str(s) for s in data_examples]
    max_data_length = max([len(s) for s in data_examples])
    data_char_processor = learn.preprocessing.ByteProcessor(max_data_length)
    x = np.array(list(data_char_processor.fit_transform(data_examples)))

    class_processor = learn.preprocessing.VocabularyProcessor(1)
    y = np.array(list(class_processor.fit_transform(class_examples)))

    return data_char_processor, class_processor

if __name__ == "__main__":
    data_file = FLAGS.data_file
    class_file = FLAGS.class_file

    print("Data File Path: ", data_file)
    print("Class File Path: ", class_file)

    data_char_processor, class_processor = load_data_labels(data_file, class_file)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    #data_char_processor.
    #class_processor.save(os.path.join(out_dir, "class"))

