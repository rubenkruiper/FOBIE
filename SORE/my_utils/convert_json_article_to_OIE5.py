import json, glob, os
from SORE.my_utils import clean_raw_input


def write_sentences_to_txt_file(input_dict, output_folder):
    """
    Reads the json input from a dataset file and prepares separate text files for OIE.

    :param input_dict: A json-file containing unprocessed papers.
    :param output_folder: Directory to write a txt file to, for each of the document IDs found in the input_dict.
    """

    processed_files = []
    for doc_id in input_dict:
        output_name = output_folder + doc_id.replace('.', '_') + '.txt'

        try:
            if os.path.exists(output_name):
                print("{} already exists, skipping and assuming it's already been processed!".format(output_name))
            else:
                content = input_dict[doc_id]
                content = clean_raw_input.clean_content(content)

                with open(output_name, 'w') as f:
                    for sent_id, sentence in content.items():
                        f.writelines(sentence['sentence']+'\n')

                processed_files.append(output_name)
                print('Processed ', doc_id, 'to a separate text file for OIE')
        except TypeError:
            print('Something wrong at: ', doc_id)



