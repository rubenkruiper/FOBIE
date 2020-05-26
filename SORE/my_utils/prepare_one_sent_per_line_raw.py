import json, glob
import sentencepiece as spm

# input_file_list = ['/media/rk22/data0/BMC_JEB/RUN_MODEL/JEB_1251-1300.json']
input_file_list = glob.glob('/media/rk22/data0/BMC_JEB/RUN_MODEL/*.json')

output_file_name = '../data/one_sent_per_line.txt'

######################
# Usage sentencepiece from python console:
# sp = spm.SentencePieceProcessor()
# spm.SentencePieceTrainer.Train('--input=one_sent_per_line.txt --model_prefix=SORE_sentencepiece_32k --vocab_size=32000 --character_coverage=1.0')

def process_json_into_single_document(input_json, output_file_name):
    with open(input_json) as input_f:
        current_file = json.load(input_f)
    #
    # def text_to_sentence(text):
    #     doc = nlp.run(text)
    #     return [s for s in doc.sents]

    list_of_sentences = []
    for doc_id in current_file:
        for sent_id in current_file[doc_id]:
            list_of_sentences += current_file[doc_id][sent_id]['sentence']+'\n'

    with open(output_file_name, 'a+') as f:
        f.writelines(list_of_sentences)


if len(input_file_list) > 1:
    total = len(input_file_list)
    for idx, input_file in enumerate(input_file_list):
        print('working on [', idx+1,'/', total,']: ', input_file)
        process_json_into_single_document(input_file, output_file_name)
else:
    print('working on [', 1, '/', 1, ']: ', input_file_list[0])
    process_json_into_single_document(input_file_list[0], output_file_name)
