import glob, os, wget
import csv, pickle, json

import numpy as np
import pandas as pd
import seaborn as sns
import sentencepiece as spm
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt

from collections import Counter
from tqdm import tqdm
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import spectral_embedding, TSNE
from sklearn.metrics.pairwise import  cosine_distances

from allennlp.commands.elmo import ElmoEmbedder

import SORE.my_utils.spacyNLP as nlp


class PrepareEmbeddings():

    def __init__(self, sp_vocab_size, csv_path, elmo_options, elmo_weights, WEIGHT_COMBINATION='max',
                 stemming=False, stopwords=False):

        self.sp_vocab_size = sp_vocab_size
        self.csv_path = csv_path
        self.WEIGHT_COMBINATION = WEIGHT_COMBINATION  # 'avg' or 'max'
        self.stemming = stemming
        self.stopwords = stopwords

        # SentencePiece
        self.sp = spm.SentencePieceProcessor()
        self.sp.load('data/filter_data/SORE_sentencepiece_{sp_vocab_size}.model'.format(sp_vocab_size=sp_vocab_size))

        ### elmo embeddings
        self.elmo = ElmoEmbedder(elmo_options, elmo_weights)

        # IDF weights
        IDF_path = "data/filter_data/subword_unit_IDF_{sp_vocab_size}.json".format(sp_vocab_size=sp_vocab_size)
        with open(IDF_path) as f:
            self.IDF_values = json.load(f)

    def stemmer(self, str_input):
        """ Porter stemmer - should actually get single words as input and output """
        self.blob = TextBlob(str_input.lower())
        tokens = self.blob.words
        stem = [token.stem() for token in tokens]
        return ''.join([w for w in stem])

    def load_data(self):

        if self.stopwords:
            stopwords = []
            with open('data/filter_data/nltk_stopwords.txt') as f:
                for line in f.readlines():
                    stopwords.append(line.strip())

        data = {}
        with open(self.csv_path, 'r') as csv_f:
            reader = csv.DictReader(csv_f)
            for row in reader:
                data[row['doc_id'] + '.' + row['sentence_nr']] = list(row['arguments'][1:-1].split(", "))


        # store all phrases by index
        all_phrases = []
        preprocessed_phrases = []
        for identifier, arguments_list in data.items():

            if arguments_list != ['']:

                arguments_list = list(set(arguments_list))

                for phrase in arguments_list:

                    all_phrases.append(phrase)

                    if self.stopwords and self.stemmer:
                        words = [t.text for t in nlp.spacy_nlp(phrase) if t.text not in stopwords]
                        preprocessed_phrase = [self.stemmer(w) for w in words]
                    elif self.stemmer:
                        words = [t.text for t in nlp.spacy_nlp(phrase)]
                        preprocessed_phrase = [self.stemmer(w) for w in words]
                    elif self.stopwords:
                        preprocessed_phrase = [t.text for t in nlp.spacy_nlp(phrase) if t.text not in stopwords]
                    else:
                        preprocessed_phrase = [t.text for t in nlp.spacy_nlp(phrase)]
                    preprocessed_phrases.append(preprocessed_phrase)

        #         print(len(all_phrases), len(preprocessed_phrases))

        return all_phrases, preprocessed_phrases


    def compute_weights_for_phrases(self, preprocessed_phrases):
        IDF_weights_for_phrases = []
        for preprocessed_phrase in preprocessed_phrases:
            IDF_weights_for_phrase = []
            for word in preprocessed_phrase:
                # initiate every subword-idx with value of 1
                subword_units = self.sp.EncodeAsIds(word)
                sw_weights = np.ones(len(subword_units))

                for sw_id, sw in enumerate(subword_units):
                    try:
                        # set index of phrase representation to corresponding IDF value
                        sw_weights[sw_id] = self.IDF_values[str(sw)]
                    except (KeyError, ValueError) as e:
                        pass
                        # No IDF value found, leave 1
                        # print("Didn't find {subword_unit_id} in IDF values: {subword_unit}".format(
                        #                subword_unit_id=sw,
                        #                subword_unit=self.sp.DecodeIds([int(sw)])))

                if len(sw_weights) < 1:
                    print(word)
                    sw_weights = [0]

                if self.WEIGHT_COMBINATION == 'max':
                    word_weight = np.amax(sw_weights)
                else:
                    word_weight = np.average(sw_weights)

                IDF_weights_for_phrase.append(word_weight)

            IDF_weights_for_phrases.append(IDF_weights_for_phrase)
        print("Prepared the IDF weights, based on subword units.")
        #         print(len(IDF_weights_for_phrases))
        return IDF_weights_for_phrases

    def embed_data(self, preprocessed_phrases):
        embeddings = []
        weights_for_phrases = self.compute_weights_for_phrases(preprocessed_phrases)
        print("Embedding phrases progress:")
        for phrase_id, phrase in enumerate(tqdm(preprocessed_phrases)):
            if len(phrase) > 0:
                elmo_embedding = self.elmo.embed_sentence(phrase)
                avg_embedding = np.average(elmo_embedding, axis=0)
                # average over words in the phrase
                weighted_embedding = np.average(avg_embedding,
                                                weights=weights_for_phrases[phrase_id],
                                                axis=0)
            else:
                # phrase is empty after pre-processing
                weighted_embedding = np.zeros([1, 1024])
            embeddings.append(weighted_embedding)
        return embeddings


class NarrowIEOpenIECombiner(object):

    def __init__(self, sp_size, csv_path,
                 number_of_clusters=50,
                 stemming=True,
                 stopwords=True,
                 weight_combination="avg",
                 path_to_embeddings=None):
        self.randomstate = 14
        self.csv_path = csv_path
        self.number_of_clusters = number_of_clusters
        self.weight_combination = weight_combination
        self.stemming = stemming
        self.stopwords = stopwords

        if sp_size in ['4k', '8k', '16k', '32k']:
            self.sp_size = sp_size
            self.IDF_path = "subword_unit_IDF_{}.json".format(sp_size)
            self.nr_of_IDF_values = int(sp_size[:-1]) * 1000
        else:
            print("Pick a sentence piece size from: '4k', '8k', '16k', '32k'")
            print("(Or create your sp model and vocab, and change code accordingly)")

        self.ELMo_options_path = ""
        self.ELMo_weights_path = ""
        if not path_to_embeddings:
            self.path_to_embeddings = "data/filter_data/elmo_pubmed/"
            self.check_for_embeddings()


    def check_for_embeddings(self):
        """
        May need to download the ELMo PubMed embeddings
        """
        embedding_files = glob.glob(self.path_to_embeddings + "*")

        if embedding_files == []:
            print('No embedding files found, beginning download of ELMo PubMed files.')
            w = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5"
            o = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            wget.download(w, self.path_to_embeddings + 'ELmo_PubMed_weights.hdf5')
            wget.download(o, self.path_to_embeddings + 'ELmo_PubMed_options.json')
            self.ELMo_weights_path = self.path_to_embeddings + 'ELmo_PubMed_weights.hdf5'
            self.ELMo_options_path = self.path_to_embeddings + 'ELmo_PubMed_options.json'
        elif self.path_to_embeddings +'ELmo_PubMed_weights.hdf5' in embedding_files:
            self.ELMo_weights_path = self.path_to_embeddings + 'ELmo_PubMed_weights.hdf5'
            self.ELMo_options_path = self.path_to_embeddings + 'ELmo_PubMed_options.json'
            print("Found ELMo PubMed embeddings")
            pass
        else:
            print("Assuming the ELMo PubMed embeddings are correctly set in {}".format(self.path_to_embeddings))
            # would have to add other types of embeddings


    def prepare_embeddings(self):
        """
        If setup doesn't exist yet, runs the embedder following user-defined settings
        and stores the output in pickle files. Otherwise, loads the existing pickle files
        for the specified settings.
        """

        settings = "{sp}_{w}_{stem}_{stop}".format(sp=self.sp_size,
                                                   w=str(self.weight_combination),
                                                   stem=str(self.stemming),
                                                   stop=str(self.stopwords))

        if not os.path.exists("data/filter_data/vectors_{settings}.pkl".format(settings=settings)):
            try:
                embedder = PrepareEmbeddings(self.sp_size,
                                             self.csv_path,
                                             self.ELMo_options_path,
                                             self.ELMo_weights_path,
                                             WEIGHT_COMBINATION=self.weight_combination,
                                             stemming=self.stemming,
                                             stopwords=self.stopwords)
                all_phrases, preprocessed_phrases = embedder.load_data()
                embeddings = embedder.embed_data(preprocessed_phrases)
            except:
                print("Narrow IE arguments not properly embedded.")
                return

            with open("data/filter_data/vectors/all_phrases_{settings}.pkl".format(settings=settings),
                      'wb') as f:
                pickle.dump(all_phrases, f)

            with open("data/filter_data/vectors/vectors_{settings}.pkl".format(settings=settings),
                      'wb') as f:
                pickle.dump(embeddings, f)
        else:
            with open("data/filter_data/vectors/all_phrases_{settings}.pkl".format(settings=settings),
                      'rb') as f:
                all_phrases = pickle.load(f)
            with open("data/filter_data/vectors/vectors_{settings}.pkl".format(settings=settings),
                      'rb') as f:
                embeddings = pickle.load(f)

        return all_phrases, embeddings


    def cluster_kmeans(self, input_matrix, KM_CLUSTERS):
        print("Starting clustering with data shape {}".format(input_matrix.shape))
        km = KMeans(n_clusters=KM_CLUSTERS, random_state=self.randomstate)
        km.fit(input_matrix)
        return km

    def clustering(self, all_phrases, embeddings):
        clustering_data = np.stack([x.squeeze() for x in embeddings])
        km_model = self.cluster_kmeans(clustering_data, self.number_of_clusters)

        distances_to_centroids = km_model.transform(clustering_data)
        distances_filtered_by_label = []
        for idx, p in enumerate(all_phrases):
            l = km_model.labels_.tolist()[idx]
            distances_filtered_by_label.append(distances_to_centroids[idx][l])

        results = pd.DataFrame()
        results['phrase'] = all_phrases
        results['category'] = km_model.labels_
        results['distance'] = distances_filtered_by_label
        print("Example of a cluster:")
        print(results.loc[results['category'] == 3])
        return km_model, results, clustering_data


    def cluster_insight(self, amount_to_print=10):
        """
        Print a couple of the clusters to see what type of arguments are found.
        """
        stopwords = []
        if self.stopwords:
            with open('data/filter_data/nltk_stopwords.txt') as f:
                for line in f.readlines():
                    stopwords.append(line.strip())

        def print_cluster_words(cluster_phrases):
            cluster_word_c = Counter()
            for phrase in cluster_phrases['phrase']:
                for token in nlp.run(phrase):
                    if token.text not in stopwords:
                        cluster_word_c[token.text] += 1
            return [w for w, c in cluster_word_c.most_common(amount_to_print)]


        for cluster_id in range(self.number_of_clusters):
            print("###################################################")
            cluster_phrases = results.loc[results['category'] == cluster_id]
            print("CLUSTER ID  - ", cluster_id,
                  "    SIZE: ", len(results.loc[results['category'] == cluster_id]))
            print(cluster_phrases.nsmallest(amount_to_print, 'distance'))

            top_phrases = cluster_phrases['phrase'].value_counts()
            print("\nTOP {} PHRASES AND COUNTS   ".format(amount_to_print))
            print(top_phrases[:amount_to_print])

            top_words = print_cluster_words(cluster_phrases)
            print("\nTOP {} WORDS IN CLUSTER".format(len(top_words)))
            print(top_words)
            print('\n\n\n')

    def scatter(x, colors, category_list, NUM_CLUSTERS):
        # We choose a color palette with seaborn.
        palette = np.array(sns.color_palette("hls", NUM_CLUSTERS))

        # We create a scatter plot.
        f = plt.figure(figsize=(60, 60))
        ax = plt.subplot()  # 'equal')
        sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=100,
                        c=palette[colors.astype(np.int)])

        ax.axis('on')
        ax.axis('tight')

        # We add the labels for each digit.
        txts = []
        for i in range(NUM_CLUSTERS):
            # Position of each label.
            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(category_list[i]), fontsize=50)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

        return f, ax, sc, txts


    def palplot(self, digits_proj, km_model, category_list):
        settings = "{sp}_{w}_{stem}_{stop}".format(sp=self.sp_size,
                                                   w=str(self.weight_combination),
                                                   stem=str(self.stemming),
                                                   stop=str(self.stopwords))

        # legend of colours
        sns.palplot(np.array(sns.color_palette("hls", self.number_of_clusters)))
        # plot
        self.scatter(digits_proj, km_model.labels_,
                category_list,
                self.number_of_clusters)
        plt.savefig('plot_{}.png'.format(settings=self.settings), dpi=120)


    def run(self,
            print_clusters=True,
            plot=True,
            cluster_names=None):

        all_phrases, embeddings = self.prepare_embeddings()
        km_model, clusters, clustering_data = self.clustering(all_phrases, embeddings)
        if print_clusters:
            self.cluster_insight(clusters)

        if cluster_names:
            # You can manually name clusters and label the clusters
            category_list = [x for x in cluster_names.values()]
        else:
            category_list = [x for x in range(self.number_of_clusters)]

        if plot:
            digits_proj = TSNE(random_state=self.randomstate).fit_transform(clustering_data)
            self.palplot(digits_proj, km_model, category_list)


csv_path = "data/narrowIE/tradeoffs_and_argmods.csv"
test = NarrowIEOpenIECombiner('8k', csv_path, 46, True, True, 'avg', None)
test.run()

