import glob, os
from pyopenie import OpenIE5
from tqdm import tqdm

"""
Change this path to the OpenIE 5 jar you will have to assemble
"""
path_to_OIE_jar = "/Users/rubenkruiper/dev/OpenIE-standalone/target/scala-2.10/openie-assembly-5.0-SNAPSHOT.jar"


oie_data_dir = 'data/OpenIE/'

class OpenIE5_client(object):
    def __init__(self, oie_data_dir, path_to_OIE_jar):
        self.oie_data_dir = oie_data_dir
        self.path_to_OIE_jar = path_to_OIE_jar
        self.current_pwd = os.getcwd()

    def determine_output_files(self):
        input_files = glob.glob(self.oie_data_dir+'inputs/*.txt')
        print(input_files)
        output_files = [self.oie_data_dir+'processed/'+ x.rsplit('/',maxsplit=1)[1][:-4] + '_processed.txt' for x in input_files]

        file_paths = []

        for idx, input_file in enumerate(input_files):
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

        input_output_files = self.determine_output_files()

        for input_file, output_file in input_output_files:

            with open(input_file, encoding='ascii', errors='ignore') as f:
                lines_in_file = f.readlines()

            number_of_lines = 0
            number_of_lines_processed = 0
            with open(output_file, 'w') as of:
                for line in tqdm(lines_in_file):
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

                print("Processed {}/{} lines in {} with OpenIE5".format(
                    number_of_lines_processed, number_of_lines, input_file[:-4].rsplit('/', maxsplit=1)[1]))

        print("Finished processing files with OpenIE5, will now shut down server.")


    def start_server(self):
        print("Starting server at port 8000")
        OIE_dir = self.path_to_OIE_jar.split('target')[0]
        os.chdir(OIE_dir)
        print("To start a server copy the following line into a new terminal window and run:")
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


def run_OpenIE_5(unprocessed_paths, path_to_OIE_jar):
    client = OpenIE5_client(unprocessed_paths, path_to_OIE_jar)
    client.start_server()
    answer = input("Wait until the server is running to continue! Is the server ready? (y, n): ").lower()
    if answer == 'y':
            client.get_extractions()
    elif answer == 'n':
            pass

    client.stop_server()

run_OpenIE_5(oie_data_dir, path_to_OIE_jar)
