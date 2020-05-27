import glob, os, sys, csv
from pyopenie import OpenIE5
from tqdm import tqdm


class OpenIE5_client(object):
    def __init__(self, csv_path, oie_data_dir, path_to_OIE_jar):
        self.csv_path = csv_path
        self.oie_data_dir = oie_data_dir
        self.path_to_OIE_jar = path_to_OIE_jar
        self.current_pwd = os.getcwd()


    def get_docid_from_filename(self, filename, output_name=False):
        if output_name:
            return self.oie_data_dir+'processed/'+filename.rsplit('/',maxsplit=1)[1][:-4]+'_processed.txt'
        return filename.rsplit('/',maxsplit=1)[1][:-4]


    def determine_in_and_output_files(self):
        input_files = glob.glob(self.oie_data_dir+'inputs/*.txt')
        docs_with_central_relations = []
        with open(self.csv_path, 'r') as csv_f:
            reader = csv.DictReader(csv_f)
            for row in reader:
                docs_with_central_relations.append(row['doc_id'])

        docs_with_central_relations = list(set(docs_with_central_relations))

        OIE_input = [f for f in input_files if self.get_docid_from_filename(f) in docs_with_central_relations]
        output_files = [self.get_docid_from_filename(x, True) for x in OIE_input]

        file_paths = []
        for idx, input_file in enumerate(OIE_input):
            file_paths.append([input_file, output_files[idx]])

        return file_paths

    def parse_extractions(self, dict_list):
        """
        Parses the OpenIE5 json output for a single line, which has the format:
        [{'confidence' : x1, 'sentence': y, 'extraction': {
                                                    'arg1': {'text': str},
                                                    'rel' : {'text': str},
                                                    'arg2s': {'text': str},
        {'confidence' : x2, 'sentence': y, 'extraction: {}' },
         etc.. ]
        """
        lines_to_write = []
        for tuple in dict_list:

            for k in tuple['extraction'].keys():
                if k not in ['arg1', 'rel', 'arg2s', 'context', 'offset', 'negated', 'passive']:
                    print("Currently not handling the OIE extraction key: {}".format(k))
                    pass

            ex =  tuple['extraction']

            line_to_write = None
            try:
                context = ""
                if ex['context']:
                    context = ex['context']['text']
                arg2s_string = ""
                for idx, arg in enumerate(ex['arg2s']):
                    arg2s_string += ", [#{}]{}".format(str(idx), arg['text'])
                line_to_write = "{:.3f}\t({},{}{})\tcontext({})\tnegated: {} ,passive: {}".format(
                    tuple['confidence'], ex['arg1']['text'],
                    ex['rel']['text'], arg2s_string,
                    context, str(ex['negated']), str(ex['passive']))
            except:
                pass

            if line_to_write:
                lines_to_write.append(line_to_write)

        return lines_to_write

    def get_extractions(self):
        os.chdir(self.current_pwd)
        extractor = OpenIE5('http://localhost:8000')

        input_output_files = self.determine_in_and_output_files()

        for input_file, output_file in input_output_files:

            if os.path.exists(output_file):
                print("{} already exists, skipping and assuming it's already been processed!".format(output_file))
            else:
                with open(input_file, encoding='ascii', errors='ignore') as f:
                    lines_in_file = f.readlines()

                number_of_lines = 0
                number_of_lines_processed = 0
                with open(output_file, 'w') as of:
                    for line in tqdm(lines_in_file, position=0, leave=True):
                        if line == '\n':
                            pass
                        else:

                            number_of_lines += 1
                            of.writelines([line.rstrip()])
                            try:
                                # extractions = os.system("curl -X POST http://localhost:8000/getExtraction -d {}".format('"'+ line + '"'))
                                extractions = extractor.extract(line.rstrip())
                                stuff_to_write = self.parse_extractions(extractions)
                                of.writelines(stuff_to_write)
                                of.writelines(['\n'])
                                number_of_lines_processed += 1

                            except:
                                print("Can't process: {}".format(line.rstrip()))
                                pass

                    sys.stderr.flush()
                    print("Processed {}/{} lines in {} with OpenIE5\n".format(
                        number_of_lines_processed, number_of_lines, input_file[:-4].rsplit('/', maxsplit=1)[1]))

        print("Finished processing files with OpenIE5, will now shut down server.")


    def start_server(self):
        print("Starting server at port 8000")
        OIE_dir = self.path_to_OIE_jar.split('target')[0]
        os.chdir(OIE_dir)
        print("To start an OpenIE5 server copy the following line into a new terminal window and run:")
        print("cd {} ; java -Xmx10g -XX:+UseConcMarkSweepGC -jar {} --ignore-errors --httpPort 8000\n".format(
            OIE_dir, self.path_to_OIE_jar
        ))

    def stop_server(self):
        try:
            os.system("kill -9 `ps aux | grep 'java -Xmx10g -XX:+UseConcMarkSweepGC -jar "
                      + self.path_to_OIE_jar +
                      " --ignore-errors --httpPort 8000'| grep -v grep | awk '{print $2; exit}'`")
            print("Stopped the server")
        except:
            print('Error shutting down a pre-existing OpenIE5 server at port 8000')


def run_OpenIE_5(csv_path,
                 path_to_OIE_jar=None,
                 unprocessed_paths='SORE/data/OpenIE/'):
    if path_to_OIE_jar == None:
        print("Change `path_to_OIE_jar` to the OpenIE 5 jar you have to assemble!")
    client = OpenIE5_client(csv_path, unprocessed_paths, path_to_OIE_jar)
    client.start_server()
    answer = input("Wait until the server is running to continue! Is the server ready? (y, n): ").lower()
    if answer == 'y':
            client.get_extractions()
    elif answer == 'n':
            pass

    client.stop_server()
