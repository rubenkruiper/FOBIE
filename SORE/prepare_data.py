import shutil, json, os
from tqdm import tqdm
from SORE.my_utils.spacyNLP import spacy_nlp
import SORE.my_utils.convert_json_article_to_OIE5 as convert_to_OIE
import SORE.my_utils.convert_json_article_to_SciIE as convert_to_SciIE


def write_dicts_to_files(num_docs, dict_with_various_docs,
                         input_doc, index, old_index,
                         output_folder_OIE, output_folder_narrowIE):
    """
    Writes the files for OIE and NarrowIE
    """
    # OIE
    convert_to_OIE.write_sentences_to_txt_file(dict_with_various_docs, output_folder_OIE)

    # NarrowIE
    if index < num_docs-1:
        narrowIE_output_name = input_doc.rsplit('/', maxsplit=1)[1]
    else:
        narrowIE_output_name = input_doc.rsplit('/', maxsplit=1)[1].replace('.',
                                '#{}-{}_narrowIE_input.'.format(old_index+1, index+1))

    output_file_path = output_folder_narrowIE + narrowIE_output_name
    if os.path.exists(output_file_path):
        print("{} already exists, skipping and assuming it's already been processed!".format(narrowIE_output_name))
    else:
        output_file = open(output_file_path, 'w', encoding='utf-8')
        narrowIE_inputdata = []
        dict_list = convert_to_SciIE.convert_doc_to_sciie_format(dict_with_various_docs)
        narrowIE_inputdata += dict_list
        for dic in dict_list:
            json.dump(dic, output_file)
            output_file.write("\n")
        print("Wrote the input for the SciIE system to: ", output_folder_narrowIE + narrowIE_output_name)


def convert_documents(max_num_docs_narrowIE, input_files, output_folder_OIE, output_folder_narrowIE):
    """
    Reads an unprocessed json file and prepares the input document for narrow and open IE. Scraped
    text in JEB and BMC files is processed to single-sentence-dict:
        # {"doc_id": {"sent_id": {"sentence":
    :param input_files: list of a json-files containing unprocessed papers
    :param output_folder_OIE: output folder for OIE files, one for each doc_id
    :param output_folder_narrowIE: output folder for NarrowIE files, one for each input_file
    """

    for input_file in input_files:
        num_docs = max_num_docs_narrowIE
        print("\nCollecting sentences from (max batch size {}): {}".format(num_docs, input_file))
        with open(input_file) as f:
            data = json.load(f)

        i = 1
        dict_with_various_docs = {}
        old_index = 0

        for index, doc_id in enumerate(tqdm(data, position=0, leave=True)):

            all_sections = data[doc_id]['sections'].keys()
            sections = []
            for section in all_sections:
                if section.lower() == 'references':
                    pass
                else:
                    sections.append(section.lower())

            # drop documents that have only one or no sections:
            if len(sections) < 2:
                print("Dropped document {}, because it only contains the sections: {}".format(doc_id, sections))
                continue
            else:
                # run in batches of 'max_num_docs_narrowIE' docs
                if (index + 1) // num_docs == i:
                    i += 1
                    # clear the dict_list
                    write_dicts_to_files(num_docs, dict_with_various_docs, input_file, index, old_index,
                                         output_folder_OIE, output_folder_narrowIE)
                    dict_with_various_docs = {}
                    old_index = index

                # add data from various documents to a single dict
                dict_with_various_docs[doc_id] = {}
                sent_count = 0

                for section in sections:
                    list_of_paragraphs = data[doc_id]['sections'][section]['text']
                    for paragraph in list_of_paragraphs:
                        parsed_paragraph = spacy_nlp(paragraph)
                        for sent in parsed_paragraph.sents:
                            if len(sent.text) > 30:
                                dict_with_various_docs[doc_id][sent_count] = {'sentence': sent.text}
                                sent_count += 1


        # process remaining docs (less than 400)
        write_dicts_to_files(num_docs, dict_with_various_docs, input_file, index, old_index,
                             output_folder_OIE, output_folder_narrowIE)

        # move the processed file, so it doesn't get processed from the start again
        processed_file = input_file.rsplit('/', maxsplit=1)[0] + "/processed/" + input_file.rsplit('/', maxsplit=1)[-1]
        shutil.move(input_file, processed_file)
        print("Processed: ", input_file.rsplit('/', maxsplit=1)[-1])


    print("Done preparing data for OIE and narrow IE!")




