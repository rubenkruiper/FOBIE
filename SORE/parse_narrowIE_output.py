import json, glob, csv
from collections import Counter



######
def convert_spans_to_tokenlist(predicted_spans, corresponding_data):
    """
    Converts the spans of relations found in a sentence to a list of tokens

    :param predicted_spans: SciIE output, formatted with span_start and span_end as token indices.
    :param corresponding_data: SciIE input file, which contains the list of tokens for each sentence.
    """
    rel_c = Counter()

    # [spans] = predicted_spans['ner']   # NER spans and RE spans do not match up!
    [relations] = predicted_spans['relation']

    all_rel_arguments = []
    tradeoff_arguments = []
    modified_tradeoff_arguments = []

    for rel in relations:
        rel_c[rel[4]] += 1

        if rel[4] == "Not_a_TradeOff":
            all_rel_arguments.append(corresponding_data["sentences"][0][rel[0]:rel[1] + 1])
            all_rel_arguments.append(corresponding_data["sentences"][0][rel[2]:rel[3] + 1])

        if rel[4] == "TradeOff":
            tradeoff_arguments.append(corresponding_data["sentences"][0][rel[0]:rel[1] + 1])
            tradeoff_arguments.append(corresponding_data["sentences"][0][rel[2]:rel[3] + 1])
            all_rel_arguments.append(corresponding_data["sentences"][0][rel[0]:rel[1] + 1])
            all_rel_arguments.append(corresponding_data["sentences"][0][rel[2]:rel[3] + 1])

        # collect arg_modifiers for trade-off relations as well, once trade-off args are known
        if rel[4] == "Arg_Modifier":
            arg_1 = corresponding_data["sentences"][0][rel[0]:rel[1] + 1]
            arg_2 = corresponding_data["sentences"][0][rel[2]:rel[3] + 1]

            if arg_1 in tradeoff_arguments:
                modified_tradeoff_arguments.append(arg_1 + arg_2)
            elif arg_2 in tradeoff_arguments:
                modified_tradeoff_arguments.append(arg_2 +  arg_1)

            if arg_1 in all_rel_arguments:
                all_rel_arguments.append(arg_2)
            elif arg_2 in all_rel_arguments:
                all_rel_arguments.append(arg_1)

    return all_rel_arguments, tradeoff_arguments, modified_tradeoff_arguments, rel_c


def simple_tokens_to_string(tokenlist):
    """
    Convert a list of tokens to a string.

    :param tokenlist: A list of tokens from the spacy parser
    :return : A string with all tokens concatenated, simply separated by a space.
    """
    return ' '.join(x for x in tokenlist if (x != '<s>' and x != '</s>'))


def read_sciie_output_format(data_doc, predictions_doc, RELATIONS_TO_STORE):
    """
    Reads the SciIE input and predictions, and prepares a list of arguments to write to a csv file. Choices for RELATIONS_TO_STORE:
      * ALL - Use all narrow IE arguments and relations found in all documents.
      * TRADEOFFS - Use all narrow IE arguments and relations found in documents where a TradeOff relation was found.
      * TRADEOFFS_AND_ARGMODS - Use only the TradeOff relations and their modifiers (in documents where a TradeOff relation was found).

    :param data_doc: the input data to the SciIE system.
    :param predictions_doc: the predictions from the SciIE system for the same input data.
    :param RELATIONS_TO_STORE: variable that determines which arguments to store  - choice
       between 'ALL', 'TRADEOFFS', and 'TRADEOFFS_AND_ARGMODS'
    :return: `output_all_sentences` a list of rows to write to a CSV file -
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

    lines_to_write = []
    all_relations_counter = Counter()

    for preds_for_sent, sent in zip(predicted_dicts, data_dicts):
        rel_args_for_sent = []
        if preds_for_sent['relation'] != [[]]:

            all_modified_args, to_args, modified_to_args, rel_counter = convert_spans_to_tokenlist(preds_for_sent, sent)
            all_relations_counter += rel_counter

            doc_id, sent_id = preds_for_sent['doc_key'].rsplit('_', maxsplit=1)
            doc_id = doc_id.replace('.','_')
            sentence = simple_tokens_to_string(sent["sentences"][0])

            if RELATIONS_TO_STORE == "ALL":
                relation_types = 'All'
                argument_list = [simple_tokens_to_string(arg) for arg in all_modified_args]
            if RELATIONS_TO_STORE == "TRADEOFFS":
                if to_args != []:
                    relation_types = 'All relations for documents with a TradeOff'
                    argument_list = [simple_tokens_to_string(arg) for arg in all_modified_args]
            if RELATIONS_TO_STORE == "TRADEOFFS_AND_ARGMODS":
                relation_types = 'Only TradeOffs with their Arg-Modifiers'
                argument_list = [simple_tokens_to_string(arg) for arg in (modified_to_args)]

            if argument_list != []:
                rel_args_for_sent.append([doc_id, sent_id, sentence, relation_types, argument_list])

        if rel_args_for_sent != []:
            lines_to_write.append(rel_args_for_sent)

    print("Relations found: ", rel_counter.most_common())

    return lines_to_write


def start_parsing(data, pred, output_csv, RELATIONS_TO_STORE):
    """
    Start the parsing of a single set of narrow IE predictions, and write these to a temporary CSV file.
    The CSV file will be combined with others into one large CSV. Choices for RELATIONS_TO_STORE:
      * ALL - Use all narrow IE arguments and relations found in all documents.
      * TRADEOFFS - Use all narrow IE arguments and relations found in documents where a TradeOff relation was found.
      * TRADEOFFS_AND_ARGMODS - Use only the TradeOff relations and their modifiers (in documents where a TradeOff relation was found).

    :param data: narrowIE input.
    :param pred: narrowIE predictions.
    :param output_csv: temporary csv file name.
    :param RELATIONS_TO_STORE: Settings for which relatiosn to store.
    :return:
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

