import json, scipy, glob
from collections import Counter
from SORE.my_utils.spacyNLP import spacy_nlp

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
import pprint

import sentencepiece as spm



def get_trade_off_arguments_from_file(sore_input, oie_input):

    with open(sore_input) as f:
        x = json.load(f)

    trade_off_arguments = []
    phrases_word_c = Counter()
    sent_tracker = {}

    for doc_id, annotations in x.items():
        if doc_id == oie_input:
            for idx, dict in enumerate(annotations):
                for rel in dict['relations']:
                    for sent_id in rel:
                        gov = rel[sent_id]['Gov']
                        dep = rel[sent_id]['Dep']
                        sent_tracker[sent_id] = []
                        if len(gov) == 1:
                            trade_off_arguments.append(gov[0])
                        else:
                            trade_off_arguments.append(' '.join(x for x in gov if x != '</s>'))
                            sent_tracker[sent_id].append(' '.join(x for x in gov if x != '</s>'))
                        if len(dep) == 1:
                            trade_off_arguments.append(dep[0])
                        else:
                            trade_off_arguments.append(' '.join(x for x in dep if x != '</s>'))


    trade_off_arguments = list(set(trade_off_arguments)) # keep only unique phrases

    return trade_off_arguments


def parse_openie_triple(line):
    if 'Context(' in line:
        without_context = line.split(':', maxsplit=1)
        parts = without_context[1].rstrip().split(';')
    else:
        parts = line.rstrip().split(';')

    try:
        arg0 = parts[0].split('(', maxsplit=1)[1]
        rel = parts[1]
        arg1 = parts[2].rsplit(')', maxsplit=1)[0]
    except IndexError:
        return "skipping line because it start with 0. and I'm too hasted to implement regex"

    return [arg0, rel, arg1]


def read_openie_results(file_name, oie_cutoff):
    all_results = open(file_name).readlines()
    doc_extractions = {}

    triple_c = 0
    unique_arg_c = Counter()
    unique_rel_c = Counter()
    oie_unique_triples_c = Counter()
    arg_len_c = Counter()

    sent_no = 0
    current_sent_triples = []
    sentence_extractions = {}

    for line in all_results:
        if line.startswith('0.') or line.startswith('1.00'):
            if float(line[:4]) < oie_cutoff:
                triple = parse_openie_triple(line)
                if type(triple) == list:
                    current_sent_triples.append(triple)
                    triple_c += 1
                    unique_arg_c[triple[0]] += 1
                    unique_rel_c[triple[1]] += 1
                    unique_arg_c[triple[2]] += 1
                    oie_unique_triples_c[tuple(triple)] += 1

                    arg_len_c[len([t for t in spacy_nlp(triple[0])])] += 1 ### NEED TO FIND THE TOKEN LENGTH!
                    arg_len_c[len([t for t in spacy_nlp(triple[2])])] += 1 ### NEED TO FIND THE TOKEN LENGTH!

                else:
                    print(triple)
                    pass
        elif line == '\n': # Assume new sentence
            sentence_extractions['extractions'] = current_sent_triples
            doc_extractions[sent_no] = sentence_extractions
            sentence_extractions = {}
            current_sent_triples = []
            sent_no += 1
        else:
            sentence_extractions['sentence'] = line

    print("OPENIE STATS UNFILTERED")
 #   print("Avg. arg_len: ", ( sum([k*c for k, c in arg_len_c.items()] ) / len(unique_arg_c.keys())))
    print("Unique args: ", len(unique_arg_c.keys()))
    print("Unique rel: ", len(unique_rel_c.keys()))
    print("Unique TRIPLES: ", len(oie_unique_triples_c.keys()))
    return doc_extractions



def get_list_of_all_arguments(sore_input, oie_input):
    all_args = []
    predicted_concepts = get_sore_arguments(sore_input)
    # predicted_concepts = get_trade_off_arguments_for_file(file, prediction_file, False)
    oie_dict = read_openie_results(oie_input)
    oie_arguments = []
    for sent_id in oie_dict:
        for triple in oie_dict[sent_id]['extractions']:
            oie_arguments.append(triple[0])
            oie_arguments.append(triple[2])

    all_args += predicted_concepts
    all_args += oie_arguments

    return all_args

# def get_oie_dict(file_path):
#     # file_path, file_name = file_path.rsplit('/', maxsplit=1) #old_stuff
#     # file_name = file_name.rsplit('.', maxsplit=1)[0].replace('.', '_') + '.txt'
#     oie_dict = read_openie_results(file_path)# +'/'+ file_name)
#     return oie_dict


def process_input_file(input_file):
    """
    Reads the filtered OpenIE5 extractions prints them to the console
    TO DO:
        - store the filtered extractions instead
        - output in format usable for graph-construction
    :param input_file:
    :return:
    """
    file_name = input_file.rsplit('_')[-1]

    total_rel_c = 0
    oie_rel_c = 0
    oie_unique_rel_c = Counter()
    oie_unique_triples_c = Counter()
    oie_unique_arg_c = Counter()
    oie_arg_len_c = Counter()

    with open(input_file) as f:
        _, filtered_triples = json.load(f).values()

    # get the trade-off
    my_list = []
    with open('data/JEB_PREDICTIONS_' + file_name) as f:
        for line in f:
            my_list.append(json.loads(line))
    trade_off_stuff_to_print = {}
    for annotations in my_list:
        gov = annotations['gov']
        dep = annotations['dep']
        rel = annotations['rel']
        sentence = tuple(annotations['sentence'])

        total_rel_c += 1

        if sentence in trade_off_stuff_to_print:
            trade_off_stuff_to_print[sentence].append(tuple([gov, '>>', rel, '>>', dep]))
        else:
            trade_off_stuff_to_print[sentence] = [tuple([gov, '>>', rel, '>>', dep])]

    # get the filtered extractions
    linked_extractions = {}
    for sent_id, triples_dict in filtered_triples.items():
        for trade_off_arg in triples_dict.keys():
            similar_oie_arg, similarity_info = 0,0
            for k,v in triples_dict[trade_off_arg].items():
                similar_oie_arg = k
                similarity_info = v
            oie_triple, cos_similarity, arg_num = similarity_info

            total_rel_c += 1
            oie_rel_c += 1
            oie_unique_triples_c[tuple(oie_triple)] += 1
            oie_unique_arg_c[oie_triple[0]] += 1
            oie_unique_rel_c[oie_triple[1]] += 1
            oie_unique_arg_c[oie_triple[2]] += 1

            oie_arg_len_c[len([t for t in nlp.run(oie_triple[0])])] += 1
            oie_arg_len_c[len([t for t in nlp.run(oie_triple[2])])] += 1

            if trade_off_arg in linked_extractions:
                linked_extractions[trade_off_arg].append(tuple([oie_triple, " --> ", arg_num, "(", round(cos_similarity, 2), ") sent: ", sent_id]))
            else:
                linked_extractions[trade_off_arg]  = [tuple([oie_triple, " --> ", arg_num, "(", round(cos_similarity, 2), ") sent: ", sent_id])]

            # print(trade_off_arg)
            # print("\t", oie_triple, " --> ", arg_num, "(", round(cos_similarity, 2), ")")

    # Can store this as a indented JSON file?
    for k, v in trade_off_stuff_to_print.items():
        print(k)
        for rel in v:
            print("\t", rel)
            # [print(r) for r in rel]

    for k, v in linked_extractions.items():
        print(k)
        for rel in v:
            print("\t", rel)
            # [print(r) for r in rel]

    print("OPENIE STATS FILTERED")
    print("Avg. arg_len: ", (sum([k * c for k, c in oie_arg_len_c.items()]) / len(oie_unique_arg_c.keys())))
    print("Unique args: ", len(oie_unique_arg_c.keys()))
    print("Unique rel: ", len(oie_unique_rel_c.keys()))
    print("Unique TRIPLES: ", len(oie_unique_triples_c.keys()))

    # print(trade_off_stuff_to_print)
    # print(linked_extractions)


    print(filtered_triples[sent_id])

    # create a list per document of the all relations
    # maybe indent the filtered extractions


# SORE_input = glob.glob("../SORE_output/*.json")
# OIE_input = glob.glob("../parsed_OIE/*.txt")

# TEST input!
SORE_input = ["../data/test_paper/SORE_output/transport_networks.json"]
OIE_input = ["../data/test_paper/parsed_OIE/transport_networks.txt"]

# run for a selection of JEB/BMC files
#file_list = ['BMC_1045', 'BMC_0190', 'JEB_2254']
#SORE_input = ["../SORE_output/"+x+".json" for x in file_list]
#OIE_input = ["../parsed_OIE/"+x+".txt" for x in file_list]


IDF_path = '../data/BMC_JEB_IDF.json'
oie_cutoff = .7             # minimum value of OPENIE score

cos_threshold = .85
overlap_threshold = 1      # max percentage of argument-overlap with respect to total nr of args that
idf_threshold = 2.
# ELMO embedding for cosine similarity:
options_file = "/media/rk22/data0/PLOSone_embeddings/data/elmo/pubmed.json"
weight_file = "/media/rk22/data0/PLOSone_embeddings/data/elmo/pubmed.hdf5"
#options_file = "../data/elmo/pubmed.json"
#weight_file = "../data/elmo/pubmed.hdf5"

################################## code ###
# model = sent2vec.Sent2vecModel()
# model.load_model('/media/rk22/data0/wiki16gb.bin')
sp = spm.SentencePieceProcessor()
sp.load('../data/SORE_sentencepiece_8k.model')


elmo = ElmoEmbedder(options_file, weight_file)
pp = pprint.PrettyPrinter(indent=4)

with open(IDF_path) as f:
    IDF_values = json.load(f)
print("Loaded IDF weights")

def check_phrase_similarity(sentence_1, weights_1, sentence_2, weights_2):
    """  elmo.embed_sentence(correlated_c.split(' '))  """
    # ELMo test
    elmo_sent_1 = elmo.embed_sentence(sentence_1)
    elmo_sent_2 = elmo.embed_sentence(sentence_2)
    avg_sent_1 = np.average(elmo_sent_1, axis=0)
    avg_sent_2 = np.average(elmo_sent_2, axis=0)
    weighted_sent_1 = np.average(avg_sent_1, weights=weights_1, axis=0)
    weighted_sent_2 = np.average(avg_sent_2, weights=weights_2, axis=0)

    # Sent2Vec test
    # s2v_sent_1 = model.embed_sentence(' '.join(w for w in sentence_1)) # (1, 700)
    # s2v_sent_2 = model.embed_sentence(' '.join(w for w in sentence_2))
    # weighted_sent_1 = np.average(s2v_sent_1, weights=weights_1, axis=0)
    # weighted_sent_2 = np.average(s2v_sent_2, weights=weights_2, axis=0)

    return scipy.spatial.distance.cosine(weighted_sent_1, weighted_sent_2)


def filter_results_for_file(oie_dict, rel_args, arg_weights):

    filtered_triples = {}

    def check_similarity_for_entire_arg(triple, idx, rel_arg):
        # Spacy tokens IDF
        # to_w = arg_weights[arg]
        rel_w = arg_weights[tuple(rel_arg)]
        oie_arg = triple[idx]
        oie_w = arg_weights[tuple(triple[idx])]
        # oie_w = arg_weights[triple[idx]]

        try:
            """ Entire arg similarity """
            cos = check_phrase_similarity(oie_arg, oie_w, rel_arg, rel_w)
            if cos > cos_threshold:
                return {str(tuple(rel_arg)):{str(tuple(oie_arg)):
                                                 str(tuple([triple, cos, 'OIE_arg_'+str(idx)]))}}
        except:
            print('Issues with:', rel_arg, ' && ', oie_arg)

    def check_similarity_per_word(triple,idx, rel_arg):
        # for every word in triple[idx], for every word in rel_arg check if something is similar:
        for oie_word in triple[idx]:
            for rel_word in rel_arg:
                oie_w = IDF_values[oie_word]
                rel_w = IDF_values[rel_word]
                cos = check_phrase_similarity([oie_word], [oie_w], [rel_word], [rel_w])
                if cos > cos_threshold:
                    return {"rel_arg:"+str(tuple(rel_arg)): {"oie_arg:"+str(tuple(triple[idx])):
                                                      str(tuple([triple, cos, 'OIE_arg_' + str(idx)]))}}

    print("Nr. of sents in doc:", len(oie_dict.keys()))
    for sent_id in oie_dict:
        print("sentence: ", sent_id)
        filtered_triples[sent_id] = {}
        for triple in oie_dict[sent_id]['extractions']:
            for arg in rel_args:
                if triple[0] != None:
                    sim_whole = check_similarity_for_entire_arg(triple, 0, arg)
                    sim_per_word = check_similarity_per_word(triple, 0, arg)
                    if sim_whole != None:
                        print("Sim WHOLE!! ", sim_whole)
                        filtered_triples[sent_id].update(sim_whole)
                    elif sim_per_word != None:
                        filtered_triples[sent_id].update(sim_per_word)
                if triple[2] != None:
                    sim_whole = check_similarity_for_entire_arg(triple, -1, arg)
                    sim_per_word = check_similarity_per_word(triple, -1, arg)
                    if sim_whole != None:
                        print("Sim WHOLE!! ", sim_whole)
                        filtered_triples[sent_id].update(sim_whole)
                    elif sim_per_word != None:
                        filtered_triples[sent_id].update(sim_per_word)

        print("Filtered triples for sent ", sent_id, ":")
        print(filtered_triples[sent_id])
    return filtered_triples


def filter_oie_arguments_and_weights(oie_dict):
    """
    For every OIE argument, prepare a weight-vector
    """
    weights_for_args_dict = {}

    def prepare_weight_vector_for_spacy_token(arg):
        arg_list = []
        weights_list = []
        for w in nlp.run(arg):
            try:
                if IDF_values[w.text] > idf_threshold:
                    arg_list.append(w.text)
                    weights_list.append(IDF_values[w.text])
            except KeyError:
                print("Removed from args, since no IDF value: ", w.text)
        if arg_list:
            weights_for_args_dict[tuple(arg_list)] = weights_list
            # return ' '.join(filtered_arg)
            # return [t.text for t in nlp.run(str(filtered_arg))]
            return arg_list

    def prepare_weights_for_sentencepiece_units(arg):
        arg_list = []
        weights_list = []
        for w_id in sp.EncodeAsIds(arg):
            try:
                if IDF_values[w_id] > idf_threshold:
                    arg_list.append(w_id)
                    weights_list.append(IDF_values[w_id])
            except KeyError:
                print("Removed from args, since no IDF value: ", w_id)
        if arg_list:
            weights_for_args_dict[tuple(arg_list)] = weights_list
            return arg_list

    for sent_id in oie_dict:
        for idx, triple in enumerate(oie_dict[sent_id]['extractions']):
            filtered_oie_args = []
            # oie_dict[sent_id]['extractions'][idx][0] = prepare_weight_vector_for_spacy_token(triple[0])
            # oie_dict[sent_id]['extractions'][idx][-1] = prepare_weight_vector_for_spacy_token(triple[-1])
            oie_dict[sent_id]['extractions'][idx][0] = prepare_weights_for_sentencepiece_units(triple[0])
            oie_dict[sent_id]['extractions'][idx][-1] = prepare_weights_for_sentencepiece_units(triple[-1])

    weights_for_args_dict[None] = 0
    return oie_dict, weights_for_args_dict


def filter_arguments_and_weights(rel_args):
    """
    For every SORE arg, prepare filtered IDF weights
    """
    filtered_rel_args = []
    filtered_rel_weights = {}
    for arg in rel_args:
        filtered_arg = []
        filtered_weights = []
        for w in nlp.run(arg):
            try:
                if IDF_values[w.text] > idf_threshold:
                    filtered_arg.append(w.text)
                    filtered_weights.append(IDF_values[w.text])
            except KeyError:
                filtered_weights.append(1)
                    # print("Why don't we find: ", w.text)
        if filtered_arg:
            # filtered_rel_args.append(' '.join(filtered_arg))
            filtered_rel_weights[tuple(filtered_arg)] = filtered_weights
            # filtered_rel_args.append([t.text for t in nlp.run(str(filtered_arg))])
            filtered_rel_args.append(filtered_arg)

    return filtered_rel_args, filtered_rel_weights


def write_to_file(sore_input, oie_dict, filtered_triples):
    # for all the filtered triples in the file store the overview in a readable format:
    base_path, directory, filename = sore_input.rsplit('/', maxsplit=2)
    output_filename = base_path + '/FILTERED_OIE/' + filename

    # NEED TO MAKE THE DICT DUMPABLE (tuples are issue) -> convert to string
    json_dict = { "oie_dict": oie_dict,
                  'filtered_triples': filtered_triples}

    with open(output_filename, 'w') as f:
        json.dump(json_dict, f)


# RUN
for sore_input, oie_input in zip(SORE_input, OIE_input):
    # Prepare the IDF weights for the file
    # list_of_arguments = get_list_of_all_arguments(sore_input, oie_input)

    # get the concepts predicted by our model for a specific file
    sore_args = get_sore_arguments(sore_input)
    # get the oie arguments in a tractable dict
    oie_args = read_openie_results(oie_input, oie_cutoff)
    oie_arguments, oie_weights = filter_oie_arguments_and_weights(oie_args)
    # filter the arguments and their weights, following the IDF threshold
    sore_arguments, sore_weights = filter_arguments_and_weights(sore_args)
    oie_weights.update(sore_weights)
    # filter the oie triples:
    filtered_triples = filter_results_for_file(oie_arguments, sore_arguments, oie_weights)
    # store in readible format
    write_to_file(sore_input, oie_arguments, filtered_triples)
