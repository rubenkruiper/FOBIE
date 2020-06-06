import json, glob, csv, os, pprint, pickle

import numpy as np
import pandas as pd
import seaborn as sns
import sentencepiece as spm
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt

from collections import Counter

from tqdm import tqdm
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise
from allennlp.commands.elmo import ElmoEmbedder

from SORE.my_utils.spacyNLP import spacy_nlp



class PrepareEmbeddings():
    """
    Encapsulates the settings, paths and functionality to prepare ELMo (PubMed) embeddings from narrow IE and Open IE extractions.
    """

    def __init__(self, prefix, sp_model_path, sp_vocab_size, IDF_path, csv_path,
                 elmo_options, elmo_weights, SUBWORD_UNIT_COMBINATION='max',
                 subwordunits=True, stemming=False, stopwords=False):
        """
        Initialise the embedding object with all required settings, so these settings can be retrieved later on.

        :param prefix: Experiment name
        :param sp_model_path: Path to the pre-trained SentencePiece model.
        :param sp_vocab_size: Size of the pre-trained SentencePiece model.
        :param IDF_path: Path to the pre-computed IDF weights.
        :param csv_path: Path to the narrow IE extractions (stored in a single csv file)
        :param elmo_options: Path to elmo options file
        :param elmo_weights: Path to elmo weights file
        :param SUBWORD_UNIT_COMBINATION: How to combine the weights for a selection of subwordunits that make up a single token.
        :param subwordunits: Boolean value that indicates whether subwordunits have been used during IDF weight creation.
        :param stemming: Boolean that determines whether keyphrases are stemmed before filtering.
        :param stopwords: Boolean that determines whether stopwords are removed from keyphrases before filtering.
        """

        self.prefix = prefix
        self.sp_vocab_size = sp_vocab_size
        self.csv_path = csv_path
        self.SUBWORD_UNIT_COMBINATION = SUBWORD_UNIT_COMBINATION  # 'avg' or 'max'
        self.stemming = stemming
        self.stopwords = stopwords

        # SentencePiece
        self.subwordunits = subwordunits
        if subwordunits:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(sp_model_path)

        ### elmo embeddings
        self.elmo = ElmoEmbedder(elmo_options, elmo_weights)

        # IDF weights
        with open(IDF_path) as f:
            self.IDF_values = json.load(f)


    def porterstemmer(self, str_input):
        """
        Porter stemmer - actually gets single words as input and output, so that the tokenisation of TextBlob and spacy aligns.

        :param str_input: string to stem
        :return: stemmed string
        """
        self.blob = TextBlob(str_input.lower())
        tokens = self.blob.words
        stem = [token.stem() for token in tokens]
        return ''.join([w for w in stem])


    def parse_argument_list(self, string_list):
        """
        Reads string of arguments (from the narrow IE extractions csv file) and outputs as list.
        :param string_list: Narrow IE CSV argument-string
        :return: Narrow IE argument list
        """
        return [x[1: -1] for x in string_list[1: -1].split(", ")]


    def load_narrowIE_data(self):
        """
        Loads the argument phrases from the narrow IE CSV file, and preprocesses them following the stemming/stopword settings.

        :return: Phrases (with/without stopwords and possibly stemmed)
        """

        if self.stopwords:
            stopwords = []
            with open('SORE/my_utils/nltk_stopwords.txt') as f:
                for line in f.readlines():
                    stopwords.append(line.strip())

        data = {}
        with open(self.csv_path, 'r') as csv_f:
            reader = csv.DictReader(csv_f)
            for row in reader:
                data[row['doc_id'] + '.' + row['sentence_nr']] = list(self.parse_argument_list(row['arguments']))

        narrowIE_data = {}
        # all_phrases = []
        # preprocessed_phrases = []
        for identifier, arguments_list in data.items():
            doc_id, sent_id = identifier.rsplit('.', maxsplit=1)

            if arguments_list != ['']:
                arguments_list = list(set(arguments_list))

                for phrase in arguments_list:
                    # all_phrases.append(phrase)

                    if self.stopwords and self.stemming:
                        words = [t.text for t in spacy_nlp(phrase) if t.text not in stopwords]
                        preprocessed_phrase = [self.porterstemmer(w) for w in words]
                    elif self.stemming:
                        words = [t.text for t in spacy_nlp(phrase)]
                        preprocessed_phrase = [self.porterstemmer(w) for w in words]
                    elif self.stopwords:
                        preprocessed_phrase = [t.text for t in spacy_nlp(phrase) if t.text not in stopwords]
                    else:
                        preprocessed_phrase = [t.text for t in spacy_nlp(phrase)]

                    # preprocessed_phrases.append(preprocessed_phrase)
                    if doc_id in narrowIE_data.keys():
                        narrowIE_data[doc_id].append(preprocessed_phrase)
                    else:
                        narrowIE_data[doc_id] = [preprocessed_phrase]
            # print(len(all_phrases), len(preprocessed_phrases))

        return  narrowIE_data #all_phrases, preprocessed_phrases


    def compute_weights_for_phrases(self, preprocessed_phrases):
        """
        Compute the IDF weights for the tokens in a list of preprocessed phrases

        :param preprocessed_phrases: list of phrases to compute IDF weights for
        :return: list of IDF weights for the pre-processed tokens in each phrase
        """
        IDF_weights_for_phrases = []
        for preprocessed_phrase in preprocessed_phrases:
            IDF_weights_for_phrase = []
            for word in preprocessed_phrase:
                # initiate every subword-idx with value of 1
                if self.subwordunits:
                    subword_units = self.sp.EncodeAsIds(word)
                    sw_weights = np.ones(len(subword_units))

                    for sw_id, sw in enumerate(subword_units):
                        try:
                            # set index of phrase representation to corresponding IDF value
                            sw_weights[sw_id] = self.IDF_values[str(sw)]
                        except (KeyError, ValueError) as e:
                            # raise e
                            # No IDF value found, leave 1
                            sw_weights[sw_id] = 1.0
                            print("Didn't find {subword_unit_id} in IDF values: {subword_unit}".format(
                                           subword_unit_id=sw,
                                           subword_unit=self.sp.DecodeIds([int(sw)])))

                    if len(sw_weights) < 1:
                        print(word)
                        sw_weights = [0]

                    if self.SUBWORD_UNIT_COMBINATION == 'max':
                        word_weight = np.amax(sw_weights)
                    else:
                        word_weight = np.average(sw_weights)

                    IDF_weights_for_phrase.append(word_weight)

                else: # no subwordunits used
                    try:
                        IDF_weights_for_phrase.append(self.IDF_values[str(word)])
                    except:
                        IDF_weights_for_phrase.append(1)
                        print('IDF weights not found for {}'.format(str(word)))

            # for every phrase, add the list of word-weights
            IDF_weights_for_phrases.append(IDF_weights_for_phrase)

        return IDF_weights_for_phrases


    def embed_all_narrowIE_phrases(self, phrase_dict):
        """
        Compute the embeddings for all the narrow IE phrases, following the preprocessing settings.

        :param phrase_dict: Dict with phrases, to identify which document the narrow IE extractions belong to
        :return: Dict with embeddings, to identify which document the embeddings belong to
        """
        embeddings_dict = {}

        for doc_id, preprocessed_phrases in phrase_dict.items():
            embeddings = []
            weights_for_phrases = self.compute_weights_for_phrases(preprocessed_phrases)

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

            embeddings_dict[doc_id] = embeddings

        return embeddings_dict


    def preprocess_and_embed_phrase_list(self, preprocessed_phrases):
        """
        Compute the embeddings for a list of OIE phrases.

        :param preprocessed_phrases: List of pre-processed phrases (e.g. no stopwords)
        :return: List of embeddings for pre-processed phrases (e.g. no stopwords)
        """
        weights_for_phrases = self.compute_weights_for_phrases(preprocessed_phrases)

        embeddings = []
        for phrase_id, phrase in enumerate(preprocessed_phrases):
            if len(phrase) > 0:
                elmo_embedding = self.elmo.embed_sentence(phrase)
                average_over_elmo = np.average(elmo_embedding, axis=0)
                # average over words in the phrase
                weighted_embedding = np.average(average_over_elmo,
                                                weights=weights_for_phrases[phrase_id],
                                                axis=0)
            else:
                # phrase is empty after pre-processing
                weighted_embedding = np.zeros([1, 1024])

            embeddings.append(weighted_embedding)

        return embeddings


class ClusterTradeOffs():
    """
    Encapsulates settings and paths for clustering of narrow IE arguments.
    """
    def __init__(self, filter_data_path, number_of_clusters, sp_size, stemming, stopwords):
        """
        Initialise with cluster settings for reuse.

        :param filter_data_path: Path to directory with filter data, default is "SORE/data/filter_data/"
        :param number_of_clusters: Number of clusters to compute - the maximum number of clusters depends on the size of the input data!
        :param sp_size: Size of the trained SentencePiece model.
        :param stemming: Boolean - whether keyphrases were stemmed before clustering.
        :param stopwords: Boolean whether stopwords were removed from keyphrases before clustering
        """
        self.randomstate = 14
        self.filter_data_path = filter_data_path
        self.number_of_clusters = number_of_clusters
        self.sp_size = sp_size
        self.stemming = stemming
        self.stopwords = stopwords


    def cluster_kmeans(self, input_matrix, KM_CLUSTERS):
        """
        Compute the K-means model.

        :param input_matrix: Embeddings x (pre-processed) phrases
        :param KM_CLUSTERS: Number of clusters
        :return: K-means model
        """
        print("Starting clustering with data shape {}".format(input_matrix.shape))
        km = KMeans(n_clusters=KM_CLUSTERS, random_state=self.randomstate)
        km.fit(input_matrix)
        return km


    def get_Kmeans_model(self, phrases_dict, embeddings_dict):
        """
        Compute or load a K-means model.

        :param phrases_dict: Phrases to compute the K-means model.
        :param embeddings_dict: Embeddings to compute the K-means model.
        :return: K-means model
        """

        # ToDo - Could add doc_id info to clusters.

        all_phrases = []
        embeddings = []

        for phrases in phrases_dict.values():
            all_phrases += phrases

        for embs in embeddings_dict.values():
            embeddings += embs

        clustering_data = np.stack([x.squeeze() for x in embeddings])

        settings = "{num}_{sp}{stem}_{stop}".format(num=str(self.number_of_clusters),
                                                  sp=self.sp_size + '_',
                                                  stem=str(self.stemming),
                                                  stop=str(self.stopwords))

        if not os.path.exists(self.filter_data_path + "vectors/km_model_{settings}.pkl".format(settings=settings)):
            print("Creating new K-means clustering model and storing it for reuse.")
            try:
                km_model = self.cluster_kmeans(clustering_data, self.number_of_clusters)
            except ValueError:
                print("You have chosen too many clusters w.r.t. the number of samples you're trying to cluster.")
                raise ValueError
            with open(self.filter_data_path + "vectors/km_model_{settings}.pkl".format(settings=settings),
                      'wb') as f:
                pickle.dump(km_model, f)
        else:
            with open(self.filter_data_path + "vectors/km_model_{settings}.pkl".format(settings=settings),
                      'rb') as f:
                km_model = pickle.load(f)

        return km_model


    def cluster(self, km_model, phrases_dict, embeddings_dict):
        """
        Use a K-means model to compute the clusters for phrases and corresponding embeddings

        :param km_model: The K-means model to use.
        :param phrases_dict: The phrases to compute clusters for.
        :param embeddings_dict: The embeddings corresponding to the phrases to compute clusters for.
        :return: clustering_data - Cluster-information for the input phrases, and results - a dataframe containing clustering info
        """
        all_phrases = []
        embeddings = []

        for phrases in phrases_dict.values():
            all_phrases += phrases

        for embs in embeddings_dict.values():
            embeddings += embs

        clustering_data = np.stack([x.squeeze() for x in embeddings])

        distances_to_centroids = km_model.transform(clustering_data)

        distances_filtered_by_label = []
        for idx, p in enumerate(all_phrases):
            l = km_model.labels_.tolist()[idx]
            distances_filtered_by_label.append(distances_to_centroids[idx][l])

        results = pd.DataFrame()
        results['phrase'] = [tuple(p) for p in all_phrases]
        results['category'] = km_model.labels_
        results['distance'] = distances_filtered_by_label
        # print("Example of a cluster:")
        # print(results.loc[results['category'] == 3])

        return clustering_data, results


    def print_cluster_words(self, cluster_phrases, amount_to_print):
        """
        Determine the words that are most commonly used in the phrases of a cluster.

        :param cluster_phrases: Phrases that belong to a cluster
        :param amount_to_print: Number of most-used words to return
        :return: The most used words...
        """
        cluster_word_c = Counter()
        for phrase in cluster_phrases['phrase']:
            for token in phrase:
                cluster_word_c[token] += 1
        return [w for w, c in cluster_word_c.most_common(amount_to_print)]


    def cluster_insight(self, results, amount_to_print=10):
        """
         Print a couple of lines (amount_to_print) per cluster to see what type of arguments they contain.

        :param results: The dataframe containing cluster-information (in terms of words and phrases and their distance to the centroid)
        :param amount_to_print: Number of 'top phrases' and 'top words' to print per cluster
        """
        stopwords = []
        if self.stopwords:
            with open('SORE/my_utils/nltk_stopwords.txt') as f:
                for line in f.readlines():
                    stopwords.append(line.strip())

        for cluster_id in range(self.number_of_clusters):
            print("###################################################")
            cluster_phrases = results.loc[results['category'] == cluster_id]
            print("CLUSTER ID  - ", cluster_id,
                  "    SIZE: ", len(results.loc[results['category'] == cluster_id]))
            print(cluster_phrases.nsmallest(amount_to_print, 'distance'))


            top_phrases = cluster_phrases['phrase'].value_counts()
            print("\nTOP {} PHRASES AND COUNTS   ".format(amount_to_print))
            print(top_phrases[:amount_to_print])

            top_words = self.print_cluster_words(cluster_phrases, amount_to_print)
            print("\nTOP {} WORDS IN CLUSTER".format(len(top_words)))
            print(top_words)
            print('\n\n\n')


    def scatter(self, x, colors, category_list, NUM_CLUSTERS):
        # Choose a color palette with seaborn.
        palette = np.array(sns.color_palette("hls", NUM_CLUSTERS))

        # Create a scatter plot.
        f = plt.figure(figsize=(60, 60))
        ax = plt.subplot()  # 'equal')
        sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=100,
                        c=palette[colors.astype(np.int)])

        ax.axis('on')
        ax.axis('tight')

        # Add the labels for each digit.
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
        settings = "{sp}{stem}_{stop}".format(sp=self.sp_size + '_',
                                                   stem=str(self.stemming),
                                                   stop=str(self.stopwords))

        # legend of colours
        sns.palplot(np.array(sns.color_palette("hls", self.number_of_clusters)))
        # plot
        self.scatter(digits_proj, km_model.labels_,
                     category_list,
                     self.number_of_clusters)

        # ToDo - check whether the cluster_plots dir exists and create if it doesn't (does not by default)
        plt.savefig('cluster_plots/plot_{}.png'.format(settings), dpi=120)
        print("Saved plot to 'cluster_plots/plot_{}.png'".format(settings))


################################################################## old stuff need to check ##

class SoreFilter():
    """
    Encapsulates all settings and paths for the SORE filtering step.
    """
    def __init__(self, OIE_path, csv_path, IDF_path, subwordunit, sp_model_path,
                 emb_weights, emb_options, filter_settings):
        """
        Initialise SORE filtering object.

        :param OIE_path: Path to Open IE extractions (directory with .txt files).
        :param csv_path: Path to narrow IE extractions (single csv file).
        :param IDF_path: Path to IDF weights file.
        :param subwordunit: Boolean - whether subword units were used.
        :param sp_model_path: Path to SentencePiece model (in case subwords units are used).
        :param emb_weights: Path to ELMo (PubMed) embedding weights.
        :param emb_options: Path to ELMo (PubMed) embedding options.
        :param filter_settings: Dict with the settings used during filtering
        """

        self.OIE_input = OIE_path
        self.csv_path = csv_path

        with open(IDF_path) as f:
            self.IDF_values = json.load(f)
            print("Got the IDF weights.")

        if subwordunit:
            self.subwordunit = subwordunit
            sp = spm.SentencePieceProcessor()
            sp.load(sp_model_path)

        self.elmo = ElmoEmbedder(emb_options, emb_weights)

        self.oie_cutoff = filter_settings['oie_cutoff']                         # minimum OpenIE confidence value
        self.sim_type = filter_settings['sim_type']                             # cosine/euclidean distance
        self.sim_threshold = filter_settings['sim_threshold']                   # minimum similarity value
        self.token_length_threshold = filter_settings['token_length_threshold'] # max length of args in nr of tokens
        self.idf_threshold = filter_settings['idf_threshold']                   # minimum overall IDF value

        self.pp = pprint.PrettyPrinter(indent=4)


    def parse_openie_triple(self, line):
        """
        Parses a line from an OIE file that contains an n-ary extraction.

        :param line: string that contains an OIE extraction (n-ary, not necessarily a triple)
        :return: OIE confidence score, list of arguments, relation, context, negation
        """
        confidence, tuple, context, neg_pass = line.split('\t')
        negation = neg_pass.split(':')[1].startswith('T')

        try:
            arg0, rest = tuple.split('[*A*]', maxsplit=1)[1].split('[*R*]')
            arg_list = [arg0]
            rel_and_arg2s = rest.split('[*A*]')
            rel = rel_and_arg2s.pop(0)
            arg2s = rel_and_arg2s
            for arg2 in arg2s:
                arg_list.append(arg2)

        except IndexError:
            # Don't think this ever occurs
            return "skipping line because it start with 0. and I'm too hasted to implement regex"

        return confidence, arg_list, rel, context, negation


    def preprocess_arguments(self, arg_list, embedder):
        """
        Get the settings from the embedder used to embed narrow IE phrases, in order to pre-process Open IE phrases.

        :param arg_list: List of Open IE argument-phrases.
        :param embedder: The embedder object used to embed narrow IE phrases.
        :return: Preprocessed phrases and their length in number of tokens
        """

        if embedder.stopwords:
            stopwords = []
            with open('SORE/my_utils/nltk_stopwords.txt') as f:
                for line in f.readlines():
                    stopwords.append(line.strip())

        preprocessed_phrases = []
        token_lenghts = []
        if arg_list != []:
            for phrase in arg_list:
                spacy_phrase = spacy_nlp(phrase)
                num_tokens_in_arg = len([t for t in spacy_phrase])
                token_lenghts.append(num_tokens_in_arg)

                if embedder.stopwords and embedder.stemming:
                    words = [t.text for t in spacy_phrase if t.text not in stopwords]
                    preprocessed_phrase = [embedder.porterstemmer(w) for w in words]
                elif embedder.stemming:
                    words = [t.text for t in spacy_phrase]
                    preprocessed_phrase = [embedder.porterstemmer(w) for w in words]
                elif embedder.stopwords:
                    preprocessed_phrase = [t.text for t in spacy_phrase if t.text not in stopwords]
                else:
                    preprocessed_phrase = [t.text for t in spacy_phrase]

                preprocessed_phrases.append(preprocessed_phrase)

        return [x for x in zip(preprocessed_phrases, token_lenghts)]


    def read_openie_results(self, file_name, oie_cutoff, embedder):
        """
        Reads all OIE extractions for a single document, and pre-processes the phrases following the narrow IE embedding settings.

        :param file_name: Open IE extractions file.
        :param oie_cutoff: Minimum confidence score for an Open IE extraction, if lower the extraction will be discarded.
        :param embedder: Embedder object that encapsulates the settings used to pre-process narrow IE argument-phrases.
        :return: A dictionary with all the Open IE extractions for a document.
        """
        all_results = open(file_name).readlines()

        doc_id = file_name.rsplit('/', maxsplit=1)[1].rsplit('_processed.txt')[0]
        doc_extractions = {doc_id: {}}

        extractions_for_sent = []
        for line in all_results:
            if line.startswith('0.') or line.startswith('1.00'):
                if float(line[:5]) > oie_cutoff:
                    confidence, args, rel, context, negation = self.parse_openie_triple(line)
                    # rel_tuple = (args[0], rel, tuple(args[1:]))
                    try:
                        extractions = {'args': [], 'rel': rel, 'context': context,
                                       'conf': confidence, 'negation': negation}
                        ### preprocess and count tokens
                        extractions['args'] = self.preprocess_arguments(args, embedder)
                        extractions_for_sent.append(extractions)
                    except:
                        print("Issue parsing: {}".format(line))
                        continue
            elif line == '\n':  # Empty lines shouldn't be encountered
                continue
            else:
                [doc_extractions[doc_id][sent_id].append(ex) for ex in extractions_for_sent]
                extractions_for_sent = []
                sent_id, sent = line[6:].split(']', maxsplit=1)
                doc_extractions[doc_id].update({sent_id: [sent]})

        return doc_extractions


    def get_weights_for_spacy_token(self, arg):
        """
        Get the IDF-weights for the words in a single OIE phrase.

        :param arg: single Open IE argument phrase
        :return: list of tokens and list of corresponding IDF weights
        """
        weights_for_args_dict = {}
        arg_list = []
        weights_list = []

        for w in spacy_nlp(arg):
            try:
                if self.IDF_values[w.text] > self.idf_threshold:
                    arg_list.append(w.text)
                    weights_list.append(self.IDF_values[w.text])
            except KeyError:
                print("Found token without an IDF value: ", w.text)

        if arg_list:
            weights_for_args_dict[tuple(arg_list)] = weights_list
            return arg_list,  weights_for_args_dict


    def get_weights_for_subwordunits(self, arg):
        """
        Get the IDF-weights for the subword-units of a single spacy token in an OIE phrase.

        :param arg: single Open IE argument phrase
        :return: list of tokens and list of corresponding IDF weights
        """
        weights_for_args_dict = {}
        arg_list = []
        weights_list = []

        for w_id in self.sp.EncodeAsIds(arg):
            try:
                if self.IDF_values[w_id] > self.idf_threshold:
                    arg_list.append(w_id)
                    weights_list.append(self.IDF_values[w_id])
            except KeyError:
                print("Found subwordunit without an IDF value: ", w_id)

        if arg_list:
            weights_for_args_dict[tuple(arg_list)] = weights_list
            return arg_list, weights_for_args_dict

    #
    # def get_weights_for_OIE_arguments(self, oie_dict):
    #     """
    #     Prepares a weight-vector for every Open IE argument-phrase.
    #     SEEMS TO NOT BE IN USE
    #     SEEMS TO NOT BE IN USE
    #     SEEMS TO NOT BE IN USE
    #
    #     :param oie_dict: Dict for a single document, contains a.o. sentence-ids and extractions for that sentence.
    #     :return: the
    #     """
    #     weights_for_args_dict = {None: 0}
    #     for sent_id in oie_dict:
    #         for idx, triple in enumerate(oie_dict[sent_id]['extractions']):
    #             if self.subwordunit:
    #                 oie_dict[sent_id]['extractions'][idx][0], w1 = self.get_weights_for_subwordunits(triple[0])
    #                 oie_dict[sent_id]['extractions'][idx][-1], w2 = self.get_weights_for_subwordunits(triple[-1])
    #                 # context as well? or list of secondary args
    #                 weights_for_args_dict.update(w1)
    #                 weights_for_args_dict.update(w2)
    #             else:
    #                 oie_dict[sent_id]['extractions'][idx][0], w1 = self.get_weights_for_spacy_token(triple[0])
    #                 oie_dict[sent_id]['extractions'][idx][-1], w2 = self.get_weights_for_spacy_token(triple[-1])
    #                 # context as well? or list of secondary args
    #                 weights_for_args_dict.update(w1)
    #                 weights_for_args_dict.update(w2)
    #
    #     return oie_dict, weights_for_args_dict


    def get_clusters_for_arguments(self, cluster_model, embeddings):
        """
        Compute the cluster_id for an argument-phrase.

        :param cluster_model: K-means model
        :param embeddings: List of embeddings for argument-phrases
        :return: List of cluster-ids per argument-phrase
        """
        clusters_for_args = []

        for embedding in embeddings:
            cluster_id = cluster_model.predict(embedding.reshape(1, -1))
            clusters_for_args.append(int(cluster_id))

        return clusters_for_args


    def phrase_similarity(self, narrowIE_embeddings, oie_arg_embedding):
        """
        Computes the similarity between two phrases, similarity measure to use can be set in SORE_settings.json file.

        :param narrowIE_embeddings: Embeddings for a phrase extracted by narrow IE
        :param oie_arg_embedding: Embeddings for a phrase extracted by Open IE
        :return: A similarity score
        """

        # ToDo - Add some more similarity options for people to play around with

        similarities = []

        for nIEarg in narrowIE_embeddings:
            try:
                if self.sim_type == 'euclidean':
                    similarities.append(pairwise.euclidean_distances(nIEarg.reshape(1, 1024),
                                                                   oie_arg_embedding.reshape(1, 1024)))
                else:
                    similarities.append(pairwise.cosine_similarity(nIEarg.reshape(1, 1024),
                                                                   oie_arg_embedding.reshape(1, 1024)))
            except:
                pass

        return max(similarities)

    #
    # def write_to_file(self, sore_input, oie_dict, filtered_triples):
    #     """
    #     Write the filtered triples to a file. Enables simply loading, e.g., if the pipeline breaks.
    #
    #     SEEMS TO BE OLD AND NOT USED
    #     SEEMS TO BE OLD AND NOT USED
    #     SEEMS TO BE OLD AND NOT USED
    #     SEEMS TO BE OLD AND NOT USED
    #
    #     :param sore_input:
    #     :param oie_dict:
    #     :param filtered_triples:
    #     :return:
    #     """
    #     # for all the filtered triples in the file store the overview in a readable format:
    #     base_path, directory, filename = sore_input.rsplit('/', maxsplit=2)
    #     output_filename = base_path + '/FILTERED_OIE/' + filename
    #
    #     # NEED TO MAKE THE DICT DUMPABLE (tuples are issue) -> convert to string
    #     json_dict = { "oie_dict": oie_dict,
    #                   'filtered_triples': filtered_triples}
    #
    #     with open(output_filename, 'w') as f:
    #         json.dump(json_dict, f)


    def start_filtering(self, output_dir, prefix, num_clusters, narrowIE_phrases, narrowIE_embeddings, embedder,
                        cluster_model, print_stats):
        """
        Script to filter all the Open IE extractions.

        :param output_dir: Path to a directory to write filtered files.
        :param prefix: Name of the experiment for re-use.
        :param num_clusters: Number of clusters (used to determine output name for re-use)
        :param narrowIE_phrases: Phrases found by narrow IE for all documents.
        :param narrowIE_embeddings: Embeddings for the phrases found by narrow IE for all documents.
        :param embedder: Embedder object, which stores the settings used to embed the narrow IE phrases.
        :param cluster_model: Clustering model trained on narrow IE phrases, in order to cluster the Open IE phrases.
        :param print_stats: Boolean - determines whether the pre- and post-filtering statistics should be printed.
        """

        all_OIE_files = glob.glob(self.OIE_input + "*.txt")
        OIE_pathnames = {}
        for pathname in all_OIE_files:
            doc_id = pathname.rsplit('/', maxsplit=1)[1].rsplit('_processed.txt')[0]
            OIE_pathnames[doc_id] = pathname

        OIE_dict = {}
        SORE_dict = {}

        for doc_id in narrowIE_embeddings.keys():
            args_already_seen = []

            # output_per_doc_id
            output_name_unfiltered = "{}{}_{}_unfiltered.json".format(output_dir, prefix, doc_id)
            output_name_filtered = "{}{}[{}]_{}.json".format(output_dir, prefix, num_clusters, doc_id)

            if os.path.exists(output_name_unfiltered):
                with open(output_name_unfiltered) as unfiltered:
                    OIE_dict.update(json.load(unfiltered))
            else:
                OIE_doc_dict = self.read_openie_results(OIE_pathnames[doc_id], 0.0, embedder)
                with open(output_name_unfiltered, 'w') as unfiltered:
                    json.dump(OIE_doc_dict, unfiltered)
                OIE_dict.update(OIE_doc_dict)

            # avoid filtering the same OIE doc multiple times if the code is stopped
            if os.path.exists(output_name_filtered):
                print("Found filtered extractions for {} and prefix '{}', loading these.".format(doc_id, prefix))
                with open(output_name_filtered) as filtered:
                    SORE_dict.update(json.load(filtered))
            else:
                print("Filtering OIE document: {}".format(doc_id))
                narrowIE_clusters = self.get_clusters_for_arguments(cluster_model, narrowIE_embeddings[doc_id])

                possible_SORE_dict = {doc_id: {'narrowIE_args': narrowIE_phrases[doc_id]}}
                total_triples = 0

                for sent_id, extractions_dict_list in OIE_dict[doc_id].items():
                    possible_sent_dict = {sent_id: [extractions_dict_list.pop(0)]}
                    found_triple = False

                    for extraction in extractions_dict_list:
                        if float(extraction['conf'][:5]) > self.oie_cutoff:
                            args_and_tokenlengths = extraction['args']
                            args = [a for (a,l) in args_and_tokenlengths if l < self.token_length_threshold]

                            OIE_embeddings = embedder.preprocess_and_embed_phrase_list(args)
                            OIE_clusters = self.get_clusters_for_arguments(cluster_model, OIE_embeddings)

                            for idx, arg in enumerate(args):
                                if arg in args_already_seen:
                                    continue
                                else:
                                    args_already_seen.append(arg)
                                # Do not consider phrases outside the clusters found in that document through narrow IE
                                if OIE_clusters[idx] in narrowIE_clusters:
                                    sim = self.phrase_similarity(narrowIE_embeddings[doc_id], OIE_embeddings[idx])
                                    if sim > self.sim_threshold:
                                        # triple should be added!
                                        match = {'extraction': extraction,
                                                  'cluster_match': OIE_clusters[idx],
                                                  'similarity_value': float(sim)}
                                        possible_sent_dict[sent_id].append(match)
                                        found_triple = True
                                        total_triples += 1
                                        # skip next args for the same relation, because the triple is already being added
                                        break
                    if found_triple:
                        possible_SORE_dict[doc_id].update(possible_sent_dict)

                with open(output_name_filtered, 'w') as out_file:
                    print("Done processing {}, retained {} OIE extractions.".format(doc_id, total_triples))
                    json.dump(possible_SORE_dict, out_file)
                SORE_dict.update(possible_SORE_dict)

        if print_stats:
            get_stats_unfiltered(OIE_dict)
            get_stats_filtered(SORE_dict)



############ stats printing functions

def get_stats_filtered(SORE_style_dict):
    """
    Print the statistics for a dictionary with OIE extractions, post-filtering.

    :param SORE_style_dict: Dict with OIE extractions AFTER filtering with SORE.
    """
    tuple_c = 0
    unique_arg_c = Counter()
    unique_rel_c = Counter()
    oie_unique_tuples_c = Counter()
    arg_len_c = Counter()

    for doc_id in SORE_style_dict.keys():

        for sent_id, extraction_list in SORE_style_dict[doc_id].items():
            if sent_id != 'narrowIE_args':
                extraction_list.pop(0)

                for extraction in extraction_list:
                    # need to handle the tuple
                    tuple_c += 1
                    rel_tuple = []

                    for arg in extraction['extraction']['args']:
                        rel_tuple.append(tuple(arg[0]))
                        unique_arg_c[tuple(arg[0])] += 1
                        arg_len_c[arg[1]] += 1

                    unique_rel_c[extraction['extraction']['rel']] += 1
                    rel_tuple.insert(1, tuple(extraction['extraction']['rel']))
                    oie_unique_tuples_c[tuple(rel_tuple)] += 1

    print("OPENIE STATS FILTERED ({} documents)".format(len(SORE_style_dict.keys())))
    #   print("Avg. arg_len: ", ( sum([k*c for k, c in arg_len_c.items()] ) / len(unique_arg_c.keys())))
    print("Unique args: ", len(unique_arg_c.keys()))
    print("Unique rel: ", len(unique_rel_c.keys()))
    print("Unique TRIPLES: ", len(oie_unique_tuples_c.keys()))


def get_stats_unfiltered(OIE_style_dict):
    """
    Print the statistics for a dictionary with OIE extractions, pre-filtering.

    :param SORE_style_dict: Dict with OIE extractions BEFORE filtering with SORE.
    """
    tuple_c = 0
    unique_arg_c = Counter()
    unique_rel_c = Counter()
    oie_unique_tuples_c = Counter()
    arg_len_c = Counter()

    for doc_id in OIE_style_dict.keys():
        for sent_id, extraction_list in OIE_style_dict[doc_id].items():
            if extraction_list != []:
                extraction_list.pop(0)
            else:
                continue

            for extraction in extraction_list:
                tuple_c += 1
                rel_tuple = []

                for arg in extraction['args']:
                    rel_tuple.append(tuple(arg[0]))
                    unique_arg_c[tuple(arg[0])] += 1
                    arg_len_c[arg[1]] += 1

                unique_rel_c[extraction['rel']] += 1
                rel_tuple.insert(1, tuple(extraction['rel']))
                oie_unique_tuples_c[tuple(rel_tuple)] += 1

    print("OPENIE STATS UNFILTERED ({} documents)".format(len(OIE_style_dict.keys())))
    #   print("Avg. arg_len: ", ( sum([k*c for k, c in arg_len_c.items()] ) / len(unique_arg_c.keys())))
    print("Unique args: ", len(unique_arg_c.keys()))
    print("Unique rel: ", len(unique_rel_c.keys()))
    print("Unique TRIPLES: ", len(oie_unique_tuples_c.keys()))

