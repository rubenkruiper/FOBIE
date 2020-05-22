import os
import json
import spacy
from spacy.matcher import Matcher

"""Change the directory to the folder with BRAT annotations"""
ann_dir = os.path.dirname(os.path.realpath(__file__)) + "/convert_to_json/"

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
    matcher.add('<s>', None, [{'ORTH': "<"}, {'ORTH': 's'}, {'ORTH': ">"}])
    matcher.add('</s>', None, [{'ORTH': "<"}, {'ORTH': '/s'}, {'ORTH': ">"}])


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


def get_annotations_from_ann_file(nlp, sentence, ann_file):
    """
    Stores the annotations from an .ann file into the following buffers, then stores
    them in our json format.
    """
    event_buffer = {}
    span_buffer = {}
    label_buffer = {}
    argmod_buffer = {}

    with open(ann_file) as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if line.startswith('E'):
                # Events (E) are trade-offs
                tradeoff_id, spans_and_slots = line.rstrip().split('\t')
                event_buffer[tradeoff_id] = {}
                slots = spans_and_slots.split(' ')
                # 2 to 6 slots of arguments + indicator
                for idx, slot in enumerate(slots):
                    ann_name, ann_id = slot.split(':')
                    event_buffer[tradeoff_id][ann_name] = ann_id
            elif line.startswith('T'):
                # Spans (T) are tokens, can be MWEs
                span_id, span_and_type, text = line.rstrip().split('\t')
                _, span_start, span_end = span_and_type.split(' ')
                span_buffer[span_id] = {'span_id': span_id,
                                        'span_start': span_start,
                                        'span_end': span_end,
                                        'text': text}
            elif line.startswith('A'):
                # Additional notes (A) 'negation' and 'confidence score' -- some annotations miss a confidence score!
                label_id, labeltype_tradeoff_id_label_value = line.rstrip().split('\t')
                label_split = labeltype_tradeoff_id_label_value.split(' ')
                label_type = ''
                tradeoff_id = ''
                label_value = ''
                if len(label_split) == 1:
                    print(label_split)
                for idx, l in enumerate(label_split):
                    if idx == 0:
                        label_type = l
                    elif idx ==1:
                        tradeoff_id = l
                    elif idx == 2:
                        label_value = l
                if label_type == 'Confidence' and label_value == '':
                    label_value = 'Low'

                if tradeoff_id not in label_buffer:
                    label_buffer[tradeoff_id] = {label_type: label_value}
                else:
                    label_buffer[tradeoff_id][label_type] = label_value
            elif line.startswith('R'):
                # Argument-modifier relations (R)
                # Always a relation between two arguments, even when the same span modifies two other spans!
                arg_mod_id, spans_and_slots = line.rstrip().split('\t')
                _, modifier, modified_arg = spans_and_slots.split(' ')
                mod_name, mod_id = modifier.split(':')
                modified_name, modified_id = modified_arg.split(':')
                argmod_buffer[arg_mod_id] = {mod_name: mod_id,
                                             modified_name: modified_id}

    # CONVERT SPANS TO SPACY TOKEN INDEXES
    doc = nlp(sentence)
    for token in doc:
        for span in span_buffer:
            start = int(span_buffer[span]['span_start'])
            end = int(span_buffer[span]['span_end'])
            if token.idx == start:
                span_buffer[span]['span_start'] = str(token.i)
            if token.idx + len(token.text) == end:
                span_buffer[span]['span_end'] = str(token.i)

    annotations = {'tradeoffs': {},
                   'modifiers': {}}

    for indicator_id, tradeoff_tuple in event_buffer.items():
        annotations['tradeoffs'][indicator_id] = {'labels': label_buffer[indicator_id]}
        for tradeoff_part, span_id in tradeoff_tuple.items():
            annotations['tradeoffs'][indicator_id][tradeoff_part] = span_buffer[span_id]

    for modifier_id, modifier_tuple in argmod_buffer.items():
        annotations['modifiers'][modifier_id] = {}
        for arg_name, span_id in modifier_tuple.items():
            annotations['modifiers'][modifier_id][arg_name] = span_buffer[span_id]

    for mod_id, modifier_args in annotations['modifiers']:
        if len(modifier_args) > 2:
            print('Mod_args > 2:', modifier_args)
    return annotations


def main():
    nlp = standalone_TradeoffWordSplitter()
    # includes the non-text annotation files, config files etc.
    all_files = [d for d in os.listdir(ann_dir)]
    data = {}

    for filename in all_files:
        if filename.endswith(".txt"):
            no_extension, _ = filename.rsplit('.', maxsplit=1)
            document_name, to_nr = no_extension.rsplit('_', maxsplit=1)
            sentence = ""
            with open(ann_dir + filename, "r") as f:
                sentence = f.read()

            if document_name in data:
                data[document_name][to_nr] = {'sentence': sentence}
            else:
                data[document_name] = {to_nr : {'sentence': sentence}}

            annotations = get_annotations_from_ann_file(nlp, sentence, ann_dir + filename[:-3]+"ann")
            data[document_name][to_nr]['annotations'] = annotations

    json.dump(data, open("dataset.json", 'w'), indent=4, sort_keys=True)


main()