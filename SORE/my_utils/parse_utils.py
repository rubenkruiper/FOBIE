import json
from SORE.my_utils.spacyNLP import spacy_nlp

def collect_sentences_from_scraped_text(input_file):
    """
    Reformats the scraped and processed JEB and BMC files to:
    # {"doc_id": {"sent_id": {"sentence":
    """


    print("Collecting sentences from: ", input_file)
    with open(input_file) as f:
        data = json.load(f)


    output_dict = {}
    for doc_id in data:
        # if doc_id == 'JEB.1293':
        output_dict[doc_id] = {}
        sent_count = 0
        # for section in data[doc_id]['sections']:
        for section in data[doc_id]['sections'].keys():
            if section.lower() == 'references':
                pass
            else:
                list_of_paragraphs = data[doc_id]['sections'][section]['text']
                for paragraph in list_of_paragraphs:
                    parsed_paragraph = spacy_nlp(paragraph)
                    for sent in parsed_paragraph.sents:
                        if len(sent.text) > 30:
                            output_dict[doc_id][sent_count] = {'sentence': sent.text}
                            sent_count += 1

    print("Processed: ", input_file.rsplit('/',maxsplit=1)[-1])
    return output_dict
