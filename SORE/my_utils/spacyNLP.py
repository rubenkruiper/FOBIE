import spacy
from spacy.matcher import Matcher



nlp = spacy.load('en_core_web_sm', disable=['ner','tagger'])
# no_parse_nlp = spacy.load('en_core_web_sm', disable=['ner','tagger','parser'])
matcher = Matcher(nlp.vocab)
matcher.add('trade-off', None, [{'ORTH': "trade"}, {'ORTH': '-'}, {'ORTH': "off"}])
matcher.add('trade-offs', None, [{'ORTH': "trade"}, {'ORTH': '-'}, {'ORTH': "offs"}])
matcher.add('Trade-off', None, [{'ORTH': "Trade"}, {'ORTH': '-'}, {'ORTH': "off"}])
matcher.add('Trade-offs', None, [{'ORTH': "Trade"}, {'ORTH': '-'}, {'ORTH': "offs"}])
matcher.add('Trade-Off', None, [{'ORTH': "Trade"}, {'ORTH': '-'}, {'ORTH': "Off"}]) # capitalised titles
matcher.add('Trade-Offs', None, [{'ORTH': "Trade"}, {'ORTH': '-'}, {'ORTH': "Offs"}])
matcher.add('parentheses', None, [{'ORTH': "("}, {}, {'ORTH': ")"}])

def quote_merger(doc):
    # this will be called on the Doc object in the spacy pipeline
    matched_spans = []
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        matched_spans.append(span)
    for span in matched_spans:  # merge into one token after collecting all matches
        span.merge()
    return doc

# nlp.tokenizer = custom_tokenizer(nlp)
nlp.add_pipe(quote_merger, first=True)  # add it right after the tokenizer
# no_parse_nlp.add_pipe(quote_merger, first=True)  # add it right after the tokenizer

def spacy_nlp(too_be_parsed):
    """
    Instantiate a Spacy nlp parser (spacy.load('en_core_web_sm', disable=['ner','tagger']), which matches a couple
    of 'trade-off' expressions as single tokens - rather than ['trade', '-', 'off'].

    :param too_be_parsed: Some string to be parsed, could be single or multiple sentences.
    :return: The space doc for that string, containing sentence and token information.
    """
    return nlp(too_be_parsed)

