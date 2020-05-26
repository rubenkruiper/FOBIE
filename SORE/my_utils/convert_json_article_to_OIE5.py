import json, glob

def write_sentences_to_txt_file(input_dict, output_folder):
    """
    Reads the json input from step 1 and prepares separate text files for OIE.
    :param input_docs: a json-files containing unprocessed papers
    :param output_folder: directory to write a txt file to, for each of the document IDs found in the input_dict
    """

    processed_files = []
    for doc_id in input_dict:
        output_name = output_folder + doc_id.replace('.', '_') + '.txt'
        content = input_dict[doc_id]
        processed_files.append(output_name)
        try:
            with open(output_name, 'w') as f:
                for sent_id, sentence in content.items():
                    f.writelines(sentence['sentence'] + '\n')
            print('Processed ', doc_id, 'to a separate text file for OIE')
        except TypeError:
            print('Something wrong at: ', doc_id)



