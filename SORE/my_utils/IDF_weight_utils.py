import json, glob, os
from SORE.my_utils.spacyNLP import spacy_nlp
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import sentencepiece as spm


class PrepIDFWeights():

    def __init__(self,
                 prefix,
                 input_file_dir,
                 output_dir,
                 SUBWORDUNIT=True,
                 STEMMING=False,
                 STOPWORDS=False):
        """   """
        if not output_dir.endswith('/'):
            print("Select a folder as output directory, make sure to end the string with '/'")
            return

        self.prefix = prefix
        self.output_dir = output_dir
        self.input_file_dir = input_file_dir
        self.STEMMING = STEMMING
        self.STOPWORDS = STOPWORDS
        self.SUBWORDUNIT = SUBWORDUNIT

        if SUBWORDUNIT:
            if self.STEMMING or self.STOPWORDS:
                print("Note: stemming/stopword-removal does not affect IDF values for subword units. "
                      "This hasn't been implemented, as it seems counter-productive w.r.t. the IDF values.")
            self.sp = spm.SentencePieceProcessor()
            self.sp_size = 10


    def new_sentencepiece_vocab(self, sp_storage_dir):
        """
        :param input: one-sentence-per-line raw corpus file. No need to run tokenise, normalise, or preprocess.
        :param model_prefix: output model name prefix, a model_prefix.model and model_prefix.vocab will be created.
        :param vocab_size: size of the vocabulary, e.g., 8000
        """
        input_files = glob.glob(self.input_file_dir + '*.txt')
        current_pwd = os.getcwd()+'/'
        input_paths = [current_pwd + x for x in input_files]

        model_prefix = self.prefix + '_' + str(self.sp_size)

        os.chdir(sp_storage_dir)
        try:
            spm.SentencePieceTrainer.train(input=input_paths,
                                       model_prefix=model_prefix,
                                       vocab_size=self.sp_size)
        except RuntimeError:
            print('The vocab size of your input documents is likely smaller that sp_size.')
            raise

        os.chdir(current_pwd)


    def txt_files_to_corpus(self, input_file):

        if self.STOPWORDS:
            list_of_stopwords = []
            with open('SORE/my_utils/nltk_stopwords.txt') as f:
                for line in f.readlines():
                    list_of_stopwords.append(line.strip())


        with open(input_file) as f:
            all_sentences = f.readlines()

        processed_list_of_sentences = []
        for sent in all_sentences:
            if self.SUBWORDUNIT:
                # SentencePiece, indexing subwordunits does not require stemming or stopword removal
                sentence = sent.encode('ascii','ignore')
                processed_list_of_sentences += [str(vocab_idx) for vocab_idx in self.sp.EncodeAsIds(sentence)]
            else:
                spacy_sent = sent
                if self.STOPWORDS and not self.STEMMING:
                    spacy_sent = [t.text for t in spacy_nlp(sent)]
                    spacy_sent = [w for w in spacy_sent if w not in list_of_stopwords]
                elif self.STEMMING and not self.STOPWORDS:
                    spacy_sent = [t.text for t in spacy_nlp(sent)]
                    stemmed_sent = []
                    for w in spacy_sent:
                        stem = ''.join(t.stem() for t in TextBlob(w).words)
                        stemmed_sent.append(stem)
                    spacy_sent = stemmed_sent
                elif self.STEMMING and self.STOPWORDS:
                    spacy_sent = [t.text for t in spacy_nlp(sent)]
                    stemmed_sent = []
                    for w in spacy_sent:
                        if w not in list_of_stopwords:
                            stem = ''.join(t.stem() for t in TextBlob(w).words)
                            stemmed_sent.append(stem)
                    spacy_sent = stemmed_sent

                spacy_sent.append("\n")
                processed_list_of_sentences += spacy_sent

        return processed_list_of_sentences


    def determine_output_name(self):
        if self.SUBWORDUNIT:
            output_name = self.output_dir + self.prefix + "IDF.json"
        else:
            if self.STEMMING and self.STOPWORDS:
                output_name = self.output_dir + self.prefix + "IDF_stemmed_no_stopwords.json"
            elif self.STEMMING:
                output_name = self.output_dir  + self.prefix + "IDF_stemmed.json"
            elif self.STOPWORDS:
                output_name = self.output_dir  + self.prefix + "IDF_no_stopwords.json"
            else:
                output_name = self.output_dir  + self.prefix + "IDF.json"

        return output_name


    def dummy_tokenizer(self, doc):
            return doc


    def get_idf(self, corpus):
        """
        Compute IDF values for a set of input documents
        """
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
            tokenizer=self.dummy_tokenizer,
            lowercase=False,
            preprocessor=self.dummy_tokenizer, vocabulary=None
        )

        vectorizer.fit_transform(corpus)
        idf_Y = vectorizer.idf_
        test_Y = dict(zip([str(x) for x in vectorizer.get_feature_names()], idf_Y))

        return test_Y


    def compute_IDF_weights(self, input_file_prefixes, sp_size, sp_storage_dir):
        """
        :param sp_size: the vocab size of
        """

        if self.SUBWORDUNIT:
            self.sp_size = sp_size
            sp_model_name = self.output_dir + "{}_{}.model".format(self.prefix, self.sp_size)
            if os.path.exists(sp_model_name):
                print("Loading existing sentencepiece model and vocab.")
                self.sp.Load(sp_model_name)
            else:
                print("Making new sentencepiece model and vocab from input files.")
                self.new_sentencepiece_vocab(sp_storage_dir)
                self.sp.Load(sp_model_name)

        corpus_list = []
        input_files = []

        for input_file_name_prefix in input_file_prefixes:
            input_files += glob.glob(self.input_file_dir + input_file_name_prefix + '*.txt')

        if len(input_files) > 1:
            total = len(input_files)
            for idx, input_file in enumerate(input_files):
                print('Combining sentences into a single corpus for IDF ({}/{}); {}'.format(
                    idx + 1, total, input_file))
                corpus_list.append(self.txt_files_to_corpus(input_file))

        IDF = self.get_idf(corpus_list)

        output_name = self.determine_output_name()
        with open(output_name, 'w') as f:
            json.dump(IDF, f)

        # print some tokens and IDF values
        value_types = 'words'
        if self.SUBWORDUNIT:
            value_types = 'subword units'

        print("Printing some IDF values, should be {}!".format(value_types))
        sanity_check = [x for x in IDF.keys()]
        for x in sanity_check[:10]:
            if self.SUBWORDUNIT:
                print(self.sp.DecodeIds([int(x)]))
            else:
                print(x)




