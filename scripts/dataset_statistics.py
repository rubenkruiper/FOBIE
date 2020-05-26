import json, spacy
from spacy.matcher import Matcher
from tqdm import tqdm
from collections import Counter


def standalone_TradeoffWordSplitter():
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    matcher.add('trade-off', None, [{'ORTH': "trade"}, {'ORTH': '-'}, {'ORTH': "off"}])
    matcher.add('trade-offs', None, [{'ORTH': "trade"}, {'ORTH': '-'}, {'ORTH': "offs"}])
    matcher.add('Trade-off', None, [{'ORTH': "Trade"}, {'ORTH': '-'}, {'ORTH': "off"}])
    matcher.add('Trade-offs', None, [{'ORTH': "Trade"}, {'ORTH': '-'}, {'ORTH': "offs"}])
    matcher.add('Trade-Off', None, [{'ORTH': "Trade"}, {'ORTH': '-'}, {'ORTH': "Off"}])  # capitalised titles
    matcher.add('Trade-Offs', None, [{'ORTH': "Trade"}, {'ORTH': '-'}, {'ORTH': "Offs"}])
    matcher.add('parentheses', None, [{'ORTH': "("}, {}, {'ORTH': ")"}])

    # matcher.add('<s>', None, [{'ORTH': "<"}, {'ORTH': 's'}, {'ORTH': ">"}])
    # matcher.add('</s>', None, [{'ORTH': "<"}, {'ORTH': '/s'}, {'ORTH': ">"}])

    def quote_merger(doc):
        matched_spans = []
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            matched_spans.append(span)
        for span in matched_spans:
            span.merge()
        return doc

    nlp.add_pipe(quote_merger, first=True)
    return nlp


def statistics_per_split(nlp, dataset):
    """
    """
    doc_count = 0

    # counters
    token_cnt = Counter()
    sentence_len_cnt = Counter()
    relation_cnt = Counter()
    keyphrase_cnt = Counter()
    trigger_cnt = Counter()
    span_cnt = Counter()
    rel_per_keyphrase = Counter()
    triggers_per_sent = []
    args_per_trigger = []
    spans_per_sent = []
    tuples_per_sent = []

    with open(dataset) as f:
        data = json.load(f)

    for source_doc_id in tqdm(data):
        doc_count += 1

        for sentence_id in data[source_doc_id]:
            unique_spans_in_sent = []
            unique_tuples_in_sent = []

            unique_sent_id = str(source_doc_id) + str(sentence_id)

            sentence: str = data[source_doc_id][sentence_id]['sentence']
            doc = nlp(sentence)
            sentence_len_cnt[len([t for t in doc])] += 1
            for token in doc:
                token_cnt[token.text] += 1

            tradeoffs = data[source_doc_id][sentence_id]['annotations']['tradeoffs']
            modifiers = data[source_doc_id][sentence_id]['annotations']['modifiers']

            for tradeoff_id in tradeoffs.keys():

                arguments = ((key, value) for key, value in tradeoffs[tradeoff_id].items() if key.startswith('Arg'))

                relation_type = "TradeOff"
                if 'Negation' in tradeoffs[tradeoff_id]['labels']:
                    relation_type = "Not_a_TradeOff"

                trigger = tradeoffs[tradeoff_id]['TO_indicator']['text']
                trigger_s = tradeoffs[tradeoff_id]['TO_indicator']['span_start']
                trigger_e = tradeoffs[tradeoff_id]['TO_indicator']['span_end']
                trigger_cnt[trigger] += 1
                unique_spans_in_sent.append(trigger)   # trigger is still a span
                span_cnt[trigger] += 1

                num_args_this_trigger = 0

                for arg_id, arg in arguments:
                    num_args_this_trigger += 1
                    tuple = [trigger_s, trigger_e, arg['span_start'], arg['span_end'], relation_type]
                    if tuple not in unique_tuples_in_sent:
                        relation_cnt[relation_type] += 1
                        unique_tuples_in_sent.append(tuple)

                    rel_per_keyphrase[arg['text']+unique_sent_id] += 1
                    keyphrase_cnt[(int(arg['span_end']) - int(arg['span_start']) + 1)] += 1
                    unique_spans_in_sent.append(arg['text'])
                    span_cnt[arg['text']] += 1

                args_per_trigger.append(num_args_this_trigger)


            for mod_id, modifier in modifiers.items():
                # mod_args = ((key, value) for key, value in modifier.items() if key.startswith('Arg'))

                relation_type = "Arg_Modifier"
                mod_arg1 = modifiers[mod_id]['Arg0']
                mod_arg2 = modifiers[mod_id]['Arg1']
                mod_tuple = [mod_arg1['span_start'], mod_arg1['span_end'],
                             mod_arg2['span_start'], mod_arg2['span_end'], relation_type]
                if mod_tuple not in unique_tuples_in_sent:
                    relation_cnt[relation_type] += 1
                    unique_tuples_in_sent.append(mod_tuple)

                for mod_arg in [mod_arg1, mod_arg2]:
                    rel_per_keyphrase[mod_arg['text']+unique_sent_id] += 1
                    if mod_arg['text'] not in unique_spans_in_sent:
                        unique_spans_in_sent.append(mod_arg['text'])
                        span_cnt[mod_arg['text']] += 1
                        keyphrase_cnt[(int(mod_arg['span_end']) - int(mod_arg['span_start']) + 1)] += 1

            # aggregate sentence statistics
            triggers_per_sent.append(len(tradeoffs.keys()))
            spans_per_sent.append(len(unique_spans_in_sent))
            tuples_per_sent.append(len(unique_tuples_in_sent))


    print("Statistics: \n {} samples found in {} documents".format(sum(sentence_len_cnt.values()), doc_count))
    print("# Sentences: {}".format(sum(sentence_len_cnt.values())))
    print("Avg. sent. length: {:.2f}".format(
        sum([k * v for k, v in sentence_len_cnt.items()]) / sum(sentence_len_cnt.values())))
    print("% of sents ≥ 25: {0:.2%}".format(
        sum([v for k, v in sentence_len_cnt.most_common() if k > 24]) / sum(sentence_len_cnt.values())))
    print("Relations:\n - Trade-Off: {}".format(relation_cnt["TradeOff"]))
    print(" - Not-a-Trade-Off: {}".format(relation_cnt["Not_a_TradeOff"]))
    print(" - Arg-Modifier: {}".format(relation_cnt["Arg_Modifier"]))
    print("Triggers: {}".format(sum(trigger_cnt.values())))
    print("Keyphrases: {}".format(sum(keyphrase_cnt.values())))
    print("Keyphrases w/ multiple relations: {}".format(
        len([v for v in rel_per_keyphrase.values() if v > 1])))
    print("Spans: {}".format(sum(span_cnt.values())))
    print("Max args/trigger: {}".format(max(args_per_trigger)))
    print("Max triggers/sent: {}".format(max(triggers_per_sent)))
    print("Max spans/sent: {}".format(max(spans_per_sent)))
    print("Max tuples/sent: {}".format(max(tuples_per_sent)))
    print("Total relations: {}".format(sum(tuples_per_sent)))
    # print("Total relations v2: {}".format(sum(relation_cnt.values())))

    return span_cnt, trigger_cnt, keyphrase_cnt, sentence_len_cnt, relation_cnt


def main():
    nlp = standalone_TradeoffWordSplitter()

    total_sent_lengths = Counter()
    unique_spans = Counter()
    unique_triggers = Counter()
    key_phrases = Counter()
    total_rel_cnt = Counter()

    input_files = ["../data/train_set.json", "../data/dev_set.json", "../data/test_set.json"]
    for input_file in input_files:

        spans, triggers, k_phrases, sent_cnt, rel_cnt = statistics_per_split(nlp, input_file)

        total_sent_lengths += sent_cnt
        unique_spans += spans
        unique_triggers += triggers
        key_phrases += k_phrases
        total_rel_cnt += rel_cnt

    print("----------------------  \n Combined over splits \n----------------------  ")
    print("Avg. sent. length: {:.2f}".format(
        sum([k * v for k, v in total_sent_lengths.items()]) / sum(total_sent_lengths.values())))
    print("% of sents ≥ 25: {0:.2%}".format(
        sum([v for k, v in total_sent_lengths.most_common() if k > 24]) / sum(total_sent_lengths.values())))
    print("Unique spans: ", len(unique_spans.keys()))
    print("Total number of spans: ", sum(unique_spans.values()))
    print("Total number of relations: ", sum(total_rel_cnt.values()))
    print("Unique triggers: ", len(unique_triggers.keys()))
    single_k_phrases = sum([v for k, v in key_phrases.items() if k == 1])
    print(
        "Single word keyphrases: {}({:.2%}) ".format(single_k_phrases, (single_k_phrases / sum(key_phrases.values()))))
    print("Avg. tokens per keyphrase: {:.2f}".format(
        sum([k*v for k, v in key_phrases.items()]) / sum(key_phrases.values())))

main()
