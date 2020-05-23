import json, spacy
from spacy.matcher import Matcher
from tqdm import tqdm


def standalone_TradeoffWordSplitter():
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    matcher.add('trade-off', None, [{'ORTH': "trade"}, {'ORTH': '-'}, {'ORTH': "off"}])
    matcher.add('trade-offs', None, [{'ORTH': "trade"}, {'ORTH': '-'}, {'ORTH': "offs"}])
    matcher.add('Trade-off', None, [{'ORTH': "Trade"}, {'ORTH': '-'}, {'ORTH': "off"}])
    matcher.add('Trade-offs', None, [{'ORTH': "Trade"}, {'ORTH': '-'}, {'ORTH': "offs"}])
    matcher.add('Trade-Off', None, [{'ORTH': "Trade"}, {'ORTH': '-'}, {'ORTH': "Off"}]) # capitalised titles
    matcher.add('Trade-Offs', None, [{'ORTH': "Trade"}, {'ORTH': '-'}, {'ORTH': "Offs"}])
    matcher.add('parentheses', None, [{'ORTH': "("}, {}, {'ORTH': ")"}])
    # matcher.add('<s>', None, [{'ORTH': "<"}, {'ORTH': 's'}, {'ORTH': ">"}])
    # matcher.add('</s>', None, [{'ORTH': "<"}, {'ORTH': '/s'}, {'ORTH': ">"}])


    def quote_merger(doc):
        # this will be called on the Doc object in the pipeline
        matched_spans = []
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            matched_spans.append(span)
        for span in matched_spans:  # merge into one token after collecting all matches
            span.merge()
        return doc

    nlp.add_pipe(quote_merger, first=True)  # add it right after the tokenizer
    return nlp



def convert_dataset_to_SCIIE(nlp, dataset):
    """"
    Convert dataset to required format for SciIE:
    line1 {   "clusters": [],
              "sentences": [["List", "of", "some", "tokens", "."]],
              "ner": [[[4, 4, "Generic"]]],
              "relations": [[[4, 4, 6, 17, "Tradeoff"]]],
              "doc_key": "XXX"}
    line2 {   " ...
    """
    # Read in the file
    with open(dataset) as f:
        data = json.load(f)

    converted_dataset = []
    doc_count = 0
    for source_doc_id in tqdm(data):

        doc_count += 1
        for sentence_id in data[source_doc_id]:
            sent_dict = {"clusters": [],
                         "doc_key": source_doc_id + "_" + sentence_id}
            ner = []
            relations = []

            sentence = data[source_doc_id][sentence_id]['sentence']
            doc = nlp(sentence)
            tradeoffs = data[source_doc_id][sentence_id]['annotations']['tradeoffs']
            modifiers = data[source_doc_id][sentence_id]['annotations']['modifiers']

            unique_arg_spans = []
            for tradeoff_id in tradeoffs.keys():
                predicate_start = int(tradeoffs[tradeoff_id]['TO_indicator']['span_start'])
                predicate_end = int(tradeoffs[tradeoff_id]['TO_indicator']['span_end'])

                relation_type = "TradeOff"
                if 'Negation' in tradeoffs[tradeoff_id]['labels']:
                    relation_type = "Not_a_TradeOff"

                arguments = ((key, value) for key, value in tradeoffs[tradeoff_id].items() if key.startswith('Arg'))
                indicator_span = [predicate_start, predicate_end, 'trigger']
                ner.append(indicator_span)

                for arg_id, arg in arguments:
                    arg_span = [int(arg['span_start']), int(arg['span_end']), 'argument']

                    if arg_span not in unique_arg_spans:
                        ner.append(arg_span)
                        unique_arg_spans.append(arg_span)

                    relations.append([indicator_span[0],
                                      indicator_span[1],
                                      arg_span[0], arg_span[1],
                                      relation_type])

                relation_type = "Arg_Modifier"
                for mod_id, modifier in modifiers.items():
                    mod_args = ((key, value) for key, value in modifier.items() if key.startswith('Arg'))

                    mod_relations = {}
                    for mod_arg_id, mod_arg in mod_args:
                        mod_relations[mod_arg_id] = [int(mod_arg['span_start']),
                                                     int(mod_arg['span_end']),
                                                     'argument']
                        if mod_relations[mod_arg_id] not in unique_arg_spans:
                            ner.append(mod_relations[mod_arg_id])
                            unique_arg_spans.append(mod_relations[mod_arg_id])

                    relations.append([mod_relations['Arg1'][0], mod_relations['Arg1'][1],
                                      mod_relations['Arg0'][0], mod_relations['Arg0'][1],
                                      relation_type])

            sent_dict['ner'] = [ner]
            sent_dict['relations'] = [relations]
            sent_dict['sentences'] = [[token.text for token in doc]]

            converted_dataset.append(sent_dict)

    print("{} of samples found in {} documents".format(len(converted_dataset), doc_count))
    return converted_dataset


def main():
    nlp = standalone_TradeoffWordSplitter()

    input_files = ["../data/train_set.json", "../data/dev_set.json", "../data/test_set.json"]
    for input_file in input_files:
        output_name = "../data/" + input_file.rsplit('/', 1)[1].rsplit('.')[0] + '_SCIIE.json'

        dic_list = convert_dataset_to_SCIIE(nlp, input_file)

        output_file = open(output_name, 'w', encoding='utf-8')
        for dic in dic_list:
            json.dump(dic, output_file)
            output_file.write("\n")

main()