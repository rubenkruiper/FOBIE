import json
from collections import Counter
import lib.my_utils.spacyNLP as nlp


def json_dataset_to_corpus(input_file):
    corpus_list = []
    with open(input_file) as f:
        all_data = json.load(f)

    doc_names = []

    for doc_id in all_data:
        doc_names.append(doc_id)
        text_in_file = []
        for sent_id in all_data[doc_id]:
            text_in_file.append(all_data[doc_id][sent_id]['sentence'])
        corpus_list.append(text_in_file)

    corpus = []
    for file in corpus_list:
        corpus.append('\n'.join(s for s in file))

    return corpus, doc_names




def get_sore_arguments(sore_input):

    rel_c = Counter()

    sore_annotations = []
    with open(sore_input) as f:
        for line in f:
            predictions =  json.loads('['+line.rstrip().replace("}{","},{")+']') # wrap in list to deal with multiple dicts
            [sore_annotations.append(p) for p in predictions]
    relation_arguments = []
    for annotations in sore_annotations:
        gov = annotations['gov']
        dep = annotations['dep']
        rel_c[annotations['rel']] += 1
        if annotations['rel'] in ['tradeoff','Tradeoff']:
            print(annotations)

        if len(gov) == 1:
            relation_arguments.append(gov[0])
        else:
            relation_arguments.append(' '.join(x for x in gov if x != '</s>'))
        if len(dep) == 1:
            relation_arguments.append(dep[0])
        else:
            relation_arguments.append(' '.join(x for x in dep if x != '</s>'))

    relation_arguments = list(set(relation_arguments)) # keep only unique phrases
    return relation_arguments


def get_trade_off_arguments_from_file2(sore_input, oie_input):

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

                    arg_len_c[len([t for t in nlp.run(triple[0])])] += 1 ### NEED TO FIND THE TOKEN LENGTH!
                    arg_len_c[len([t for t in nlp.run(triple[2])])] += 1 ### NEED TO FIND THE TOKEN LENGTH!

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
