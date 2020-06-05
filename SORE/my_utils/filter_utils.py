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

    def __init__(self, prefix, sp_model_path, sp_vocab_size, IDF_path, csv_path,
                 elmo_options, elmo_weights, SUBWORD_UNIT_COMBINATION='max',
                 subwordunits=True, stemming=False, stopwords=False):

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
        """ Porter stemmer - should actually get single words as input and output """
        self.blob = TextBlob(str_input.lower())
        tokens = self.blob.words
        stem = [token.stem() for token in tokens]
        return ''.join([w for w in stem])


    def parse_argument_list(self, string_list):
        return [x[1: -1] for x in string_list[1: -1].split(", ")]


    def load_narrowIE_data(self):

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
    def __init__(self, filter_data_path, number_of_clusters, sp_size, stemming, stopwords):
        self.randomstate = 14
        self.filter_data_path = filter_data_path
        self.number_of_clusters = number_of_clusters
        self.sp_size = sp_size
        self.stemming = stemming
        self.stopwords = stopwords


    def cluster_kmeans(self, input_matrix, KM_CLUSTERS):
        print("Starting clustering with data shape {}".format(input_matrix.shape))
        km = KMeans(n_clusters=KM_CLUSTERS, random_state=self.randomstate)
        km.fit(input_matrix)
        return km


    def get_Kmeans_model(self, phrases_dict, embeddings_dict):
        """
        Could add doc_id info to clusters.
        """
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
        cluster_word_c = Counter()
        for phrase in cluster_phrases['phrase']:
            for token in phrase:
                cluster_word_c[token] += 1
        return [w for w, c in cluster_word_c.most_common(amount_to_print)]


    def cluster_insight(self, results, amount_to_print=10):
        """
        Print a couple of the clusters to see what type of arguments are found.
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
        settings = "{sp}{stem}_{stop}".format(sp=self.sp_size + '_',
                                                   stem=str(self.stemming),
                                                   stop=str(self.stopwords))

        # legend of colours
        sns.palplot(np.array(sns.color_palette("hls", self.number_of_clusters)))
        # plot
        self.scatter(digits_proj, km_model.labels_,
                     category_list,
                     self.number_of_clusters)
        plt.savefig('cluster_plots/plot_{}.png'.format(settings), dpi=120)
        print("Saved plot to 'cluster_plots/plot_{}.png'".format(settings))


################################################################## old stuff need to check ##

class SoreFilter():
    def __init__(self, OIE_path, csv_path, IDF_path, subwordunit, sp_model_path,
                 emb_weights, emb_options, filter_settings):

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
        Parses a line from an OIE file that contains a triple
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
            return "skipping line because it start with 0. and I'm too hasted to implement regex"

        return confidence, arg_list, rel, context, negation


    def preprocess_arguments(self, arg_list, embedder):

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
        Reads all OIE results for a single document, and preprocesses the phrases extracted by
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


    def get_weights_for_OIE_arguments(self, oie_dict):
        """
        For every OIE argument, prepare a weight-vector
        """
        weights_for_args_dict = {None: 0}
        for sent_id in oie_dict:
            for idx, triple in enumerate(oie_dict[sent_id]['extractions']):
                if self.subwordunit:
                    oie_dict[sent_id]['extractions'][idx][0], w1 = self.get_weights_for_subwordunits(triple[0])
                    oie_dict[sent_id]['extractions'][idx][-1], w2 = self.get_weights_for_subwordunits(triple[-1])
                    # context as well? or list of secondary args
                    weights_for_args_dict.update(w1)
                    weights_for_args_dict.update(w2)
                else:
                    oie_dict[sent_id]['extractions'][idx][0], w1 = self.get_weights_for_spacy_token(triple[0])
                    oie_dict[sent_id]['extractions'][idx][-1], w2 = self.get_weights_for_spacy_token(triple[-1])
                    # context as well? or list of secondary args
                    weights_for_args_dict.update(w1)
                    weights_for_args_dict.update(w2)

        return oie_dict, weights_for_args_dict


    def get_clusters_for_arguments(self, cluster_model, embeddings):
        """
        For every narrowIE arg, determine the cluster id
        """
        clusters_for_args = []

        for embedding in embeddings:
            cluster_id = cluster_model.predict(embedding.reshape(1, -1))
            clusters_for_args.append(int(cluster_id))

        return clusters_for_args


    def phrase_similarity(self, narrowIE_embeddings, oie_arg_embedding):
        """  Might add some more similarity options, although cosine seemed best so far.  """
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


    def write_to_file(self, sore_input, oie_dict, filtered_triples):
        # for all the filtered triples in the file store the overview in a readable format:
        base_path, directory, filename = sore_input.rsplit('/', maxsplit=2)
        output_filename = base_path + '/FILTERED_OIE/' + filename

        # NEED TO MAKE THE DICT DUMPABLE (tuples are issue) -> convert to string
        json_dict = { "oie_dict": oie_dict,
                      'filtered_triples': filtered_triples}

        with open(output_filename, 'w') as f:
            json.dump(json_dict, f)


    def start_filtering(self, output_dir, prefix, num_clusters, narrowIE_phrases, narrowIE_embeddings, embedder,
                        cluster_model, print_stats):

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
                OIE_doc_dict = OIE_dict[doc_id]
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

                for sent_id, extractions_dict_list in OIE_doc_dict[doc_id].items():
                    possible_sent_dict = {sent_id: [extractions_dict_list.pop(0)]}
                    found_triple = False

                    for extraction in extractions_dict_list:
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
    Print the statistics for a dictionary with OIE extractions, pre- or post-filtering.
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
    Print the statistics for a dictionary with OIE extractions, pre- or post-filtering.
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

