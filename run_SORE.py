import glob
from SORE import prepare_data, run_OIE5, parse_narrowIE_output, filterOIE_with_narrowIE
from SORE.my_utils import IDF_weight_utils


class DataPreparation():
    def __init__(self,
                 unprocessed_data_path="SORE/data/unprocessed_data/",
                 output_folder_narrowIE='SORE/data/narrowIE/input/',
                 output_folder_OIE='SORE/data/OpenIE/inputs/'):
        self.unprocessed_data_path = unprocessed_data_path
        self.output_folder_narrowIE = output_folder_narrowIE
        self.output_folder_OIE = output_folder_OIE

    def start(self):
        input_files = glob.glob(self.unprocessed_data_path +'*.json')
        prepare_data.convert_documents(input_files, self.output_folder_OIE, self.output_folder_narrowIE)



class NarrowIEParser():
    def __init__(self,
                 input_filename, predictions_filename,
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

        self.narrowIE_data = narrowIE_data_dir + input_filename
        self.narrowIE_predictions = narrowIE_predictions_dir + predictions_filename
        self.relations_to_store = RELATIONS_TO_STORE

        # Choice between 'ALL', 'TRADEOFFS', and 'TRADEOFFS_AND_ARGMODS'
        if RELATIONS_TO_STORE == "ALL":
            self.output_csv = output_csv_path + input_filename.rsplit('.', maxsplit=1)[0] + "_all_arguments.csv"
        if RELATIONS_TO_STORE == "TRADEOFFS":
            self.output_csv = output_csv_path + input_filename.rsplit('.', maxsplit=1)[0] + "_tradeoffs.csv"
        if RELATIONS_TO_STORE == "TRADEOFFS_AND_ARGMODS":
            self.output_csv = output_csv_path + input_filename.rsplit('.', maxsplit=1)[0] + "_tradeoffs_and_argmods.csv"


    def start(self):
        parse_narrowIE_output.start_parsing(self.narrowIE_data,
                                            self.narrowIE_predictions,
                                            self.output_csv, self.relations_to_store)


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
              sp_size="8k",
              SUBWORDUNIT=True,
              STEMMING=False,
              STOPWORDS=False):
        """
        :param sp_size: string that determines the name of the sentencepiece vocabulary
        :param sp_size: string that determines the size of the sentencepiece vocabulary
        """
        if self.input_file_dir == 'SORE/data/OpenIE/inputs/':
            print("Will compute IDF weights and subword info for all document in the folder `SORE/data/OpenIE/inputs/`")
            answer = input("Continue? (y, n): ").lower()
            if answer == 'y':
                pass
            else:
                return

        IDF_computer = IDF_weight_utils.PrepIDFWeights(prefix, self.input_file_dir, self.output_dir,
                                                                        SUBWORDUNIT, STEMMING, STOPWORDS)
        IDF_computer.compute_IDF_weights(sp_size, self.output_dir)


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
                   print_clusters=True,
                   plot=False,
                   cluster_names=None,
                   print_stats=False)


class FilterCheck():
    def __init__(self):
        pass


def test():
    ### Prepare data for OpenIE and narrow IE
    # Possible to define own data directories
    prep_obj = DataPreparation()
    # prep_obj.start()


    ### Parse narrow IE (need to run yourself for now)
    # Need to define own path_to_OIE_jar after building OpenIE5
    input_filename = "more_examples.json"
    predictions_filename = "predictions_more_examples.json"

    narrowIE_parser = NarrowIEParser(input_filename, predictions_filename)
    # narrowIE_parser.start()


    ### Run Open IE
    # Need to define own path_to_OIE_jar after building OpenIE5
    path_to_OIE_jar = "/Users/rubenkruiper/dev/OpenIE-standalone/target/scala-2.10/openie-assembly-5.0-SNAPSHOT.jar"
    # run_OIE5.run_OpenIE_5(narrowIE_parser.output_csv, path_to_OIE_jar)


    # Compute IDF weights (and sentencepiece model and vocab if required)
    # settings for preparing
    sp_size = 200
    SUBWORDUNIT = True
    STEMMING = False
    STOPWORDS = True
    prefix = 'test'

    prepper = FilterPrep()
    IDF_weights_path = prepper.determine_output_name(prefix, SUBWORDUNIT, STEMMING, STOPWORDS)
    # prepper.start(prefix, sp_size, SUBWORDUNIT, STEMMING, STOPWORDS)


    # Filter!
    # settings for preparing
    oie_data_dir = 'SORE/data/OpenIE/processed/'
    sore_output_dir = 'SORE/data/processed_data/'
    sp_size = 200
    number_of_clusters = 3
    SUBWORD_UNIT_COMBINATION = "avg"
    print_stats = True
    filter_settings = {
        'oie_cutoff': .7,                       # minimum OpenIE confidence value
        'sim_type': 'cos',                      # use cosine/euclidean distance
        'sim_threshold': .85,                   # minimum similarity value
        'token_length_threshold': 25,           # max length of arg in number of tokens
        'idf_threshold': 2.,                    # minimum idf weight for a phrase
    }

    my_SORE_filter = SORE_filter(narrowIE_parser.output_csv, sore_output_dir)
    my_SORE_filter.start(prefix, filter_settings, IDF_weights_path, SUBWORDUNIT, oie_data_dir, sp_size, number_of_clusters,
                         STEMMING, STOPWORDS, SUBWORD_UNIT_COMBINATION, print_stats)

if __name__ == "__main__":
    test()

