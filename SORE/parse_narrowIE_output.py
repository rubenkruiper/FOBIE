import json, glob, csv
from collections import Counter

"""
converts the outputs of the SciIE model to a csv format for clustering.
{"doc_key": "XXX", 
 "ner": [[[8, 8, "Generic"], [10, 10, "Generic"], [12, 14, "Generic"]]], 
 "relation": [[[8, 8, 10, 10, "Not_a_TradeOff"], [8, 8, 12, 14, "Not_a_TradeOff"]]]}
"""

narrowIE_data = "data/narrowIE/example_input.json"
narrowIE_predictions = "data/narrowIE/predictions_example_input.json"

# set the GLOBAL variable - choice between 'ALL', 'TRADEOFFS', and 'TRADEOFFS_AND_ARGMODS'
RELATIONS_TO_STORE = "TRADEOFFS_AND_ARGMODS"
output_csv = "data/narrowIE/tradeoffs_and_argmods.csv"


######
def convert_spans_to_tokenlist(predicted_spans, corresponding_data):
    """
    Converts the spans to a list of tokens
    :param predicted_spans: SciIE output, formatted with span_start and span_end as token indices
    :param corresponding_data: SciIE input file, which contains the list of tokens for each sentence
    """
    rel_c = Counter()

    # [spans] = predicted_spans['ner']   # NER spans and RE spans do not match up!
    [relations] = predicted_spans['relation']

    all_arguments = []
    tradeoff_arguments = []
    modifier_arguments = []

    for rel in relations:
        rel_c[rel[4]] += 1

        all_arguments.append(corresponding_data["sentences"][0][rel[0]:rel[1] + 1])
        all_arguments.append(corresponding_data["sentences"][0][rel[2]:rel[3] + 1])

        if rel[4] == "TradeOff":
            tradeoff_arguments.append(corresponding_data["sentences"][0][rel[0]:rel[1] + 1])
            tradeoff_arguments.append(corresponding_data["sentences"][0][rel[2]:rel[3] + 1])

        # collect arg_modifiers for trade-off relations as well, once trade-off args are known
        if rel[4] == "Arg_Modifier":
            arg_1 = corresponding_data["sentences"][0][rel[0]:rel[1] + 1]
            arg_2 = corresponding_data["sentences"][0][rel[2]:rel[3] + 1]
            # only store if one of the arguments is a trade-off arg
            if arg_1 in tradeoff_arguments:
                modifier_arguments.append(arg_2)
            if arg_2 in tradeoff_arguments:
                modifier_arguments.append(arg_1)

    return all_arguments, tradeoff_arguments, modifier_arguments, rel_c


def simple_tokens_to_string(tokenlist):
    """
    Convert a list of tokens to a string
    :param tokenlist: a json-files containing unprocessed papers
    :return : a string with all tokens concatenated, simply separated by a space
    """
    return ' '.join(x for x in tokenlist if (x != '<s>' and x != '</s>'))


def read_sciie_output_format(data_doc, predictions_doc, RELATIONS_TO_STORE):
    """
    Reads the SciIE input and predictions, and prepares a list of arguments to write to a csv file.
    :param data_doc: the input data to the SciIE system
    :param predictions_doc: the predictions from the SciIE system
    :param RELATIONS_TO_STORE: variable that determines which arguments to store  - choice
                               between 'ALL', 'TRADEOFFS', and 'TRADEOFFS_AND_ARGMODS'
    :return output_all_sentences: a list of rows to write to a CSV file -
                                  [doc_id, sent_id, RELATIONS_TO_STORE, argument_list, sentence]
    """
    predicted_dicts = []
    with open(predictions_doc) as o:
        for line in o.read().split('\n'):
            if len(line) > 10:  # handle empty lines in the output_doc
                predicted_dicts.append(json.loads(line))

    data_dicts = []
    with open(data_doc) as d:
        for line in d.read().split('\n'):
            if len(line) > 10:
                data_dicts.append(json.loads(line))

    output_all_sentences = []
    for sample in predicted_dicts:
        doc_key = sample['doc_key']
        output_per_sentence = []
        for d_sample in data_dicts:
            if d_sample['doc_key'] == doc_key:
                all_pred_args, to_pred_args, mod_pred_args, rel_counter = convert_spans_to_tokenlist(sample, d_sample)

                doc_id, sent_id = doc_key.rsplit('_', maxsplit=1)
                sentence = simple_tokens_to_string(d_sample["sentences"][0])

                if RELATIONS_TO_STORE == "ALL":
                    relation_types = 'All'
                    argument_list = [simple_tokens_to_string(arg) for arg in all_pred_args]
                if RELATIONS_TO_STORE == "TRADEOFFS":
                    relation_types = 'TradeOffs'
                    argument_list = [simple_tokens_to_string(arg) for arg in to_pred_args]
                if RELATIONS_TO_STORE == "TRADEOFFS_AND_ARGMODS":
                    relation_types = 'TradeOffs with Arg-Modifiers'
                    argument_list = [simple_tokens_to_string(arg) for arg in (to_pred_args + mod_pred_args)]

                output_per_sentence.append([doc_id, sent_id, sentence, relation_types, argument_list])

        output_all_sentences.append(output_per_sentence)

    return output_all_sentences


def main(data, pred, output_csv, RELATIONS_TO_STORE):
    """
    """
    rows_to_write = read_sciie_output_format(data, pred, RELATIONS_TO_STORE)

    with open(output_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "sentence_nr", "sentence", "relation_types", "arguments"])
        for rows_per_doc in rows_to_write:
            for rows_per_sent in rows_per_doc:
                writer.writerow(rows_per_sent)
    f.close()
    print("Converted the predicted ", RELATIONS_TO_STORE, " to a csv file: ", output_csv)


main(narrowIE_data, narrowIE_predictions, output_csv, RELATIONS_TO_STORE)