import re

def clean_dict(content):
    """
    Simple cleaning of the sentences found in the input files. Is called twice, during creation of
    OIE and narrowIE files.

    :param content: a dict containing {sent_id : sentence}
    :return content: a dict containing {sent_id : sentence}, where the sentences have been cleaned
    """
    new_content = {}
    new_sent = ""
    new_sent_id = ""

    for sent_id, sent_ in content.items():

        try:
            sent = sent_['sentence'].rstrip()
            sent = sent.replace('\n', " ").replace('\t', " ")
            sent = re.sub(' +', ' ', sent)

            # # drop sentences shorter than 10 characters
            # if len(sent) < 10:
            #     continue

            # assume these are separate sentences after all in case the next sent starts with a capital
            if new_sent != '' and sent[0].isupper():
                new_content.update({new_sent_id[:-1]: {'sentence': new_sent + '.'}})
                new_sent = ''
                new_sent_id = ''

            # simply join 'possibly' broken sentences
            if sent[-1] != '.' or sent.endswith('Fig.'):
                new_sent += " " + sent
                new_sent_id += str(sent_id) + '+'
                continue

            new_sent += sent
            new_sent_id += str(sent_id)
            new_content.update({new_sent_id: {'sentence': new_sent}})
            new_sent = ''
            new_sent_id = ''

        except:
            # drop sentences that throw an error, break here to see what type of error that may be
            pass

    return new_content