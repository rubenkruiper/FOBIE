import json, glob
from SORE.my_utils.spacyNLP import spacy_nlp
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from functools import partial

import sentencepiece as spm

# input_file_list = ['/media/rk22/data0/BMC_JEB/RUN_MODEL/JEB_1251-1300.json','/media/rk22/data0/BMC_JEB/RUN_MODEL/BMC_1251-1300.json']
input_file_list = glob.glob('/media/rk22/data0/BMC_JEB/RUN_MODEL/*.json')

SUBWORDUNIT = False
stemming = True
stopwords = True

######################
if stemming and stopwords:
    output_name = "token_IDF_stemmed_stopwords"
elif stemming:
    output_name = "token_IDF_stemmed"
elif stopwords:
    output_name = "token_IDF_stopwords"
else:
    output_name = "token_IDF"

if stopwords:
    list_of_stopwords = []
    with open('.SORE/my_utils/nltk_stopwords.txt') as f:
        for line in f.readlines():
            list_of_stopwords.append(line.strip())

if SUBWORDUNIT:
    output_name = "vocab_size_blabla"
    sp = spm.SentencePieceProcessor()
    sp.Load('../../data/SORE_sentencepiece_4k.model')

def dummy_tokenizer(doc):
    return doc

no_tokenizer = partial(dummy_tokenizer)

def json_dataset_to_corpus(input_file, SUBWORDUNIT):
    corpus_list = []
    with open(input_file) as f:
        all_data = json.load(f)

    for doc_id in all_data:
        sents = []
        for sent_id in all_data[doc_id]:

            if SUBWORDUNIT: ## SentencePiece subword units;
                sentence = all_data[doc_id][sent_id]['sentence'].encode('ascii', 'ignore')  # remove unicode characters
                [sents.append(str(vocab_idx)) for vocab_idx in sp.EncodeAsIds(sentence)]
            else:
                sentence = all_data[doc_id][sent_id]['sentence']
                sent = [t.text for t in spacy_nlp.run(sentence)]
                if stopwords:
                    sent = [w for w in sent if w not in list_of_stopwords]

                if stemming:
                    stemmed_sent = []
                    for w in sent:
                        stem = ''.join(t.stem() for t in TextBlob(w).words)
                        stemmed_sent.append(stem)
                    sent = stemmed_sent
                sent.append('\n')
                sents += sent
                # sents.append(' '.join(w for w in sent) + '\n')

        corpus_list.append(sents)

    return corpus_list


def get_idf(corpus):

    vectorizer = TfidfVectorizer(
        strip_accents='unicode',
        use_idf=True,
        norm=None,
        smooth_idf=True,
        sublinear_tf=False,
        binary=False,
        # min_df=1, max_df=1.0, max_features=None, ngram_range=(1, 1),
        stop_words=None,
        analyzer='word',
        tokenizer=dummy_tokenizer,
        lowercase=False,
        preprocessor=dummy_tokenizer, vocabulary=None
    )

    vectorizer.fit_transform(corpus)
    idf_Y = vectorizer.idf_
    test_Y  = dict(zip([str(x) for x in vectorizer.get_feature_names()], idf_Y))
    return test_Y




#####
corpus_list = []
if len(input_file_list) > 1:
    total = len(input_file_list)
    for idx, input_file in enumerate(input_file_list):
        print('working on [', idx+1,'/', total,']: ', input_file)
        corpus_list += json_dataset_to_corpus(input_file, SUBWORDUNIT)

IDF = get_idf(corpus_list)

with open('../../data/{}.json'.format(output_name), 'w') as f:
    json.dump(IDF, f)

# print some tokens and IDF values
sanity_check = [x for x in IDF.keys()]
for x in sanity_check[:10]:
    if SUBWORDUNIT:
        print(sp.DecodeIds([int(x)]))
    else:
        print(x)