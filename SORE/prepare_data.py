import glob, json
from tqdm import tqdm
from SORE.my_utils.parse_utils import collect_sentences_from_scraped_text
import SORE.my_utils.convert_json_article_to_OIE5 as convert_to_OIE
import SORE.my_utils.convert_json_article_to_SciIE as convert_to_SciIE

input_files = glob.glob('data/unprocessed_data/*.json')
output_narrowIE_name = 'example_input.json'

output_folder_narrowIE = 'data/narrowIE/'
output_folder_OIE = 'data/OpenIE/inputs/'

def convert_documents(input_files, output_folder_OIE, output_folder_narrowIE):
    """
    Reads an unprocessed json file and prepares the input document for narrow and open IE
    :param input_docs: a json-files containing unprocessed papers
    """
    narrowIE_inputdata = []
    for input_doc in tqdm(input_files):
        sentence_dict = collect_sentences_from_scraped_text(input_doc)
        convert_to_OIE.write_sentences_to_txt_file(sentence_dict, output_folder_OIE)
        dict_list = convert_to_SciIE.convert_doc_to_sciie_format(sentence_dict)
        narrowIE_inputdata += dict_list

    output_file = open(output_folder_narrowIE + output_narrowIE_name, 'w', encoding='utf-8')
    for dic in narrowIE_inputdata:
        json.dump(dic, output_file)
        output_file.write("\n")

    print("Wrote the input for the SciIE system to: ", output_folder_narrowIE + output_narrowIE_name)
    print("Done!")


convert_documents(input_files, output_folder_OIE, output_folder_narrowIE)