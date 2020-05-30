from SORE.my_utils.spacyNLP import spacy_nlp
from SORE.my_utils import clean_raw_input

def convert_doc_to_sciie_format(input_dict):
    """
    Reads an unprocessed json file and prepares a list of sentences in the SciIE format
    :param input_docs: a json-files containing unprocessed papers
    :return: processed_files: list of sentences ready to be input to a trained SciIE model
    """
    processed_sentences = []
    for doc_id in input_dict:

        content = input_dict[doc_id]
        content = clean_raw_input.clean_content(content)

        for sent_id, sentence in content.items():
            sent_dict = {"clusters": [],
                         "doc_key": doc_id + "_" + str(sent_id)}
            doc = spacy_nlp(sentence['sentence'])
            sent_dict['ner'] = [[]]
            sent_dict['relations'] = [[]]
            sent_dict['sentences'] = [[token.text for token in doc]]
            processed_sentences.append(sent_dict)

    return processed_sentences

