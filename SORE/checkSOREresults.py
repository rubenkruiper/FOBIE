import json
from collections import Counter
from SORE.my_utils.spacyNLP import spacy_nlp as nlp

test_file = 'FILTERED_OIE/JEB_1285.json'


def process_input_file(input_file):
    """
    Reads the filtered OpenIE5 extractions prints them to the console
    TO DO:
        - store the filtered extractions instead
        - output in format usable for graph-construction --> triples?
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




process_input_file(test_file)