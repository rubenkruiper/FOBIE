import tensorflow as tf
import tensorflow_hub as hub
import h5py
import numpy as np
import json

print tf.__version__
from util import set_gpus


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
                                sentences: sent,
                                text_len: slen
                            }
                        )
                        sentence_id = docid + '_' + str(j)
                        ds = fout.create_dataset(
                            sentence_id, shape=lm_emb.shape[1:], dtype='float32',
                            data=lm_emb[0, :, :, :]  # [slen, lm_size, lm_layers]
                        )
                fout.close


train_input = './data/processed_data/json/train_set_SCIIE.json'
train_embeddings = './data/processed_data/elmo/train_set_SCIIE.hdf5'
training_set_embedder = EmbedAndPredict()
training_set_embedder.Elmo(train_input, train_embeddings)
          
dev_input = './data/processed_data/json/dev_set_SCIIE.json'
dev_embeddings = './data/processed_data/elmo/dev_set_SCIIE.hdf5'
dev_set_embedder = EmbedAndPredict()
dev_set_embedder.Elmo(dev_input, dev_embeddings)

test_input = './data/processed_data/json/test_set_SCIIE.json'
test_embeddings = './data/processed_data/elmo/test_set_SCIIE.hdf5'
test_set_embedder = EmbedAndPredict()
test_set_embedder.Elmo(test_input, test_embeddings)