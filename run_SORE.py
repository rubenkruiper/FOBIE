import glob, json, os
import pandas as pd
from SORE import prepare_data, run_OIE5, parse_narrowIE_output, filterOIE_with_narrowIE, SORE_to_BRAT
from SORE.my_utils import IDF_weight_utils


class DataPreparation():
    def __init__(self,
                 unprocessed_data_path="SORE/data/unprocessed_data/",
                 output_folder_narrowIE='SORE/data/narrowIE/input/',
                 output_folder_OIE='SORE/data/OpenIE/inputs/'):
        self.unprocessed_data_path = unprocessed_data_path
        self.output_folder_narrowIE = output_folder_narrowIE
        self.output_folder_OIE = output_folder_OIE

    def start(self, max_num_docs_narrowIE):
        input_files = glob.glob(self.unprocessed_data_path +'*.json')
        prepare_data.convert_documents(max_num_docs_narrowIE, input_files, self.output_folder_OIE, self.output_folder_narrowIE)


class NarrowIEParser():
    def __init__(self,
                 narrowIE_data_dir="SORE/data/narrowIE/input/",
                 narrowIE_predictions_dir = "SORE/data/narrowIE/predictions/",
                 RELATIONS_TO_STORE = "TRADEOFFS_AND_ARGMODS",
                 output_csv_path = "SORE/data/narrowIE/"):
        """

        converts the outputs of the SciIE model to a csv format for clustering.
        {"doc_key": "XXX",
         "ner": [[[8, 8, "Generic"], [10, 10, "Generic"], [12, 14, "Generic"]]],
         "relation": [[[8, 8, 10, 10, "Not_a_TradeOff"], [8, 8, 12, 14, "Not_a_TradeOff"]]]}
        """
        self.narrowIE_data = narrowIE_data_dir
        self.narrowIE_predictions = narrowIE_predictions_dir
        self.relations_to_store = RELATIONS_TO_STORE
        self.output_csv_path = output_csv_path


    def start(self, input_filename, predictions_filename, output_csv):
        parse_narrowIE_output.start_parsing(self.narrowIE_data + input_filename,
                                            self.narrowIE_predictions + predictions_filename,
                                            output_csv, self.relations_to_store)


class FilterPrep():
    def __init__(self,
                 input_file_dir='SORE/data/OpenIE/inputs/',
                 output_dir="SORE/data/filter_data/"):
        """
        Prepares the subword and IDF weights
        :param input_file_dir: Directory with .txt files to process. Has to be one sentence a line.
        :param output_dir: Directory to store IDF weights and sentencepiece models & vocabs
        """
        self.input_file_dir = input_file_dir
        self.output_dir = output_dir

    def determine_output_name(self,
                              prefix='test',
                              SUBWORDUNIT=True,
                              STEMMING=False,
                              STOPWORDS=False):
        if SUBWORDUNIT:
            output_name = self.output_dir + prefix + "IDF.json"
        elif STEMMING and STOPWORDS:
            output_name = self.output_dir + prefix + "IDF_stemmed_no_stopwords.json"
        elif STEMMING:
            output_name = self.output_dir  + prefix + "IDF_stemmed.json"
        elif STOPWORDS:
            output_name = self.output_dir  + prefix + "IDF_no_stopwords.json"
        else:
            output_name = self.output_dir  + prefix + "IDF.json"

        return output_name


    def start(self,
              prefix='test',
              file_names="OA-STM",
              sp_size="8k",
              SUBWORDUNIT=True,
              STEMMING=False,
              STOPWORDS=False):
        """
        :param sp_size: string that determines the name of the sentencepiece vocabulary
        :param sp_size: string that determines the size of the sentencepiece vocabulary
        """
        if self.input_file_dir == 'SORE/data/OpenIE/inputs/':
            print("Compute IDF weights (and subword model) for all documents in the folder `SORE/data/OpenIE/inputs/`",
                  " that starts with either {}".format(str(file_names)))
            answer = input("Continue? (y, n): ").lower()
            if answer == 'y':
                pass
            else:
                return

        IDF_computer = IDF_weight_utils.PrepIDFWeights(prefix, self.input_file_dir, self.output_dir,
                                                                        SUBWORDUNIT, STEMMING, STOPWORDS)
        IDF_computer.compute_IDF_weights(file_names, sp_size, self.output_dir)


class SORE_filter():
    def __init__(self, csv_path="data/narrowIE/tradeoffs_and_argmods.csv",
                 sore_output_dir="SORE/data/processed_data/"):
        self.csv_path = csv_path
        self.sore_output_dir = sore_output_dir

    def start(self, prefix, filter_settings, IDF_weights_path, SUBWORDUNIT,
                 oie_data_dir='SORE/data/OpenIE/processed/',
                 sp_size=16000,
                 number_of_clusters=50,
                 stemming=False,
                 stopwords=True,
                 SUBWORD_UNIT_COMBINATION="avg",
                 print_stats=False,
                 path_to_embeddings=None):
        """
        :param prefix: A name to identify the created files that store IDFweights, sentencepiece model and vocab, etc.
        :param IDF_weights_path: Path to the IDF_weights created during filter preparation.
        :param SUBWORDUNIT: Boolean value that indicates whether subwordunits have been used during IDF weight creation.
        :param sp_size: Size of vocab used for sentencepiece subwordunits.
        :param stemming: Boolean that determines whether keyphrases are stemmed before filtering.
        :param stopwords: Boolean that determines whether stopwords are removed from keyphrases before filtering.
        :param SUBWORD_UNIT_COMBINATION: How the weights for subwordunits are combined to a single weight per word.
        :param print_stats: Whether to print the statistics on unfiltered OIE extractions.
        :param path_to_embeddings: Path where ELMo PubMed embeddings can be found.
        """
        filter = filterOIE_with_narrowIE.NarrowIEOpenIECombiner(oie_data_dir,IDF_weights_path, self.csv_path, SUBWORDUNIT, sp_size,
                                                                number_of_clusters, stemming, stopwords,
                                                                SUBWORD_UNIT_COMBINATION, path_to_embeddings)

        filter.run(prefix, filter_settings, self.sore_output_dir,
                   print_stats,
                   print_clusters=True,
                   plot=False,
                   cluster_names=None)


def main(all_settings):
    oie_data_dir = 'SORE/data/OpenIE/processed/'
    sore_output_dir = 'SORE/data/processed_data/'
    brat_output_dir = 'SORE/data/brat_annotations/'

    prep = all_settings['Prepare_data']
    parse_narrow = all_settings['Parse_narrowIE_predictions']
    runOIE = all_settings['Run_OIE']
    filter = all_settings['Filter_OIE']

    max_num_docs_narrowIE = all_settings['data_prep']['max_num_docs_narrowIE']

    narrowIE_input_files = all_settings['narrowIE']['narrowIE_input_files']
    RELATIONS_TO_STORE = all_settings['narrowIE']['RELATIONS_TO_STORE']


    path_to_OIE_jar = all_settings['OpenIE']['path_to_OIE_jar']

    sp_size = all_settings['Filtering']['sp_size']
    SUBWORDUNIT = all_settings['Filtering']['SUBWORDUNIT']
    STEMMING = all_settings['Filtering']['STEMMING']
    STOPWORDS = all_settings['Filtering']['STOPWORDS']
    prefix = all_settings['Filtering']['prefix']
    file_names = all_settings['Filtering']['file_names']
    number_of_clusters = all_settings['Filtering']['number_of_clusters']
    SUBWORD_UNIT_COMBINATION = all_settings['Filtering']['SUBWORD_UNIT_COMBINATION']
    print_stats = all_settings['Filtering']['print_stats']
    filter_settings = all_settings['Filtering']['filter_settings']

    convert_back_to_BRAT = all_settings['convert_back_to_BRAT']

    if prep:
        ### Prepare data for OpenIE and narrow IE
        # Possible to define own data directories
        prep_obj = DataPreparation()
        prep_obj.start(max_num_docs_narrowIE)

    # determine name for parsed narrowIE files
    suffix_options = ['ALL', 'TRADEOFFS', 'TRADEOFFS_AND_ARGMODS']
    suffixes = ["_all_arguments.csv", "_tradeoffs.csv", "_tradeoffs_and_argmods.csv"]
    output_suffix = suffixes[suffix_options.index(RELATIONS_TO_STORE)]
    narrowIE_parser = NarrowIEParser(RELATIONS_TO_STORE=RELATIONS_TO_STORE)
    combined_name = narrowIE_parser.output_csv_path + prefix + output_suffix

    if parse_narrow:
        ### Parse narrow IE (need to run yourself for now)
        parse_files = [[input_filename, "predictions_"+input_filename] for input_filename in narrowIE_input_files]
        csv_files = []

        for input_filename, predictions_filename in parse_files:
            output_csv = narrowIE_parser.output_csv_path + input_filename.rsplit('.', maxsplit=1)[0] + output_suffix
            narrowIE_parser.start(input_filename, predictions_filename, output_csv)
            csv_files.append(output_csv)

        combined_csv = pd.concat([pd.read_csv(f, engine='python') for f in csv_files])
        combined_csv.to_csv(combined_name, index=False, encoding='utf-8')
        print("Written all predictions to {}.".format(combined_name))

        for file_to_remove in csv_files:
            os.remove(file_to_remove)

    if runOIE:
        ### Run Open IE
        # If run separately from narrowOIE, the csv_path has to be
        # Need to define own path_to_OIE_jar after building OpenIE5
        run_OIE5.run_OpenIE_5(combined_name, path_to_OIE_jar)

    prepper = FilterPrep()
    my_SORE_filter = SORE_filter(combined_name, sore_output_dir)
    if filter:
        # Compute IDF weights (and sentencepiece model and vocab if required)
        # settings for preparing

        IDF_weights_path = prepper.determine_output_name(prefix, SUBWORDUNIT, STEMMING, STOPWORDS)
        if os.path.exists(IDF_weights_path):
            print("Assuming IDF weights and sentencepiece model exist, since path exists: {} ".format(IDF_weights_path))
        else:
            prepper.start(prefix, file_names, sp_size, SUBWORDUNIT, STEMMING, STOPWORDS)

        # Filter!
        my_SORE_filter.start(prefix, filter_settings, IDF_weights_path, SUBWORDUNIT, oie_data_dir, sp_size,
                             number_of_clusters, STEMMING, STOPWORDS, SUBWORD_UNIT_COMBINATION, print_stats)

    if convert_back_to_BRAT:
        dataset_paths = ['SORE/data/unprocessed_data/processed/' + d for d in narrowIE_input_files]
        converter = SORE_to_BRAT.BratConverter(dataset_paths, combined_name, sore_output_dir, brat_output_dir)
        converter.convert_to_BRAT(prefix)


if __name__ == "__main__":
    with open("SORE/SORE_settings.json") as f:
        all_settings = json.load(f)

    main(all_settings)

