#!/usr/bin/env python

import h5py, json, glob, os
import re, sys, time, random

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import util
from util import set_gpus
from lsgn_data import LSGNData
from lsgn_evaluator_writer import LSGNEvaluator
from srl_model import SRLModel


sys.path.append(os.getcwd())
# print tf.__version__


class EmbedAndPredict():
    def __init__(self):
        #### Embedding Model #####
        set_gpus(0)
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
        self.sentences = tf.placeholder('string', shape=(None, None))
        self.text_len = tf.placeholder('int32', shape=(None))

        lm_embeddings = self.elmo(
            inputs={
                "tokens": self.sentences,
                "sequence_len": self.text_len
            },
            signature="tokens", as_dict=True)

        word_emb = tf.expand_dims(lm_embeddings["word_emb"], 3)  # [B, slen, 512]
        self.lm_emb_op = tf.concat([
            tf.concat([word_emb, word_emb], 2),  # [B, slen, 1024, 1]
            tf.expand_dims(lm_embeddings["lstm_outputs1"], 3),
            tf.expand_dims(lm_embeddings["lstm_outputs2"], 3)], 3)  # [B, slen, 1024, 3]
        

    def Elmo(self, fn, outfn):
        with open(fn) as f:
            dev_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        sents = [example["sentences"] for example in dev_examples]
        docids = [example["doc_key"] for example in dev_examples]

        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            with h5py.File(outfn) as fout:
                # for line in fin:
                for i in range(len(sents)):
                    if i % 100 == 0:
                        print
                        'Finished ' + str(i)
                    doc = sents[i]
                    docid = docids[i]
                    for j in range(len(doc)):
                        sent = [doc[j]]
                        slen = [len(doc[j])]
                        lm_emb = sess.run(
                            self.lm_emb_op, feed_dict={
                                self.sentences: sent,
                                self.text_len: slen
                            }
                        )
                        sentence_id = docid + '_' + str(j)
                        ds = fout.create_dataset(
                            sentence_id, shape=lm_emb.shape[1:], dtype='float32',
                            data=lm_emb[0, :, :, :]  # [slen, lm_size, lm_layers]
                        )
                fout.close



    def write_single(self, input_json, input_hdf5):

        if len(sys.argv) > 1:
            name = sys.argv[1]
            print("Running experiment: {} (from command-line argument).".format(name))
        else:
            name = os.environ["EXP"]
            print("Running experiment: {} (from environment variable).".format(name))

        config = util.get_config("experiments.conf")[name]
        config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

        config["batch_size"] = -1
        config["max_tokens_per_batch"] = -1

        # Override config
        # lm_path_dev = "./data/processed_data/elmo/FILE_TO_RUN.hdf5"
        # eval_path = "./data/processed_data/json/FILE_TO_RUN.json"
        # output_path = "./FOBIE_output/FILE_TO_RUN.json"
        file_name = input_json.rsplit('/', 1)[1][:-5]
        output_path = "./FOBIE_output/predictions_{}.json".format(file_name)
        config["lm_path"] = input_hdf5
        config["eval_path"] = input_json
        config["output_path"] = output_path


        util.print_config(config)
        data = LSGNData(config)
        model = SRLModel(data, config)
        evaluator = LSGNEvaluator(config)

        variables_to_restore = []
        for var in tf.global_variables():
            print(var.name)
            if "module/" not in var.name:
                variables_to_restore.append(var)

        saver = tf.train.Saver(variables_to_restore)
        log_dir = config["log_dir"]

        with tf.Session() as session:
            checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
            print("Evaluating {}".format(checkpoint_path))
            tf.global_variables_initializer().run()
            saver.restore(session, checkpoint_path)
            evaluator.evaluate(session, data, model.predictions, model.loss)

        return output_path


######## code to check which files to run over #######################
paths_to_files = glob.glob('./data/processed_data/json/*.json')
paths_to_embeddings = glob.glob('./data/processed_data/elmo/*.hdf5')
embeddings_file_names = [em.rsplit('/', 1)[1][:-5] for em in paths_to_embeddings]
files_to_process = []
for file_path in paths_to_files:
    file_name = file_path.rsplit('/', 1)[1][:-5]
    answer = raw_input("Generate ELMo embeddings and predict for {}? (y, n): ".format(file_name))
    if answer == 'y':
        files_to_process.append(file_name)
    else:
        pass

# run for each selected input file
for file_name in files_to_process:
    input_json = './data/processed_data/json/{}.json'.format(file_name)
    input_hdf5 = './data/processed_data/elmo/{}.hdf5'.format(file_name)

    embedder_predictor = EmbedAndPredict()

    # if embeddings don't exist, generate
    if os.path.exists(input_hdf5) == False:
        embedder_predictor.Elmo(input_json, input_hdf5)

    # predict
    output_path = embedder_predictor.write_single(input_json, input_hdf5)

    # remove embeddings again
    if os.path.exists(output_path):
        print("Could have removed {}".format(input_hdf5))
        # print("Removing the embeddings, since they get very big.")
        # os.remove(input_hdf5)
