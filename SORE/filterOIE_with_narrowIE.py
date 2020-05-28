import glob, os, wget
import csv, pickle

from sklearn.manifold import TSNE
import SORE.my_utils.filter_utils as fu



class NarrowIEOpenIECombiner(object):

    def __init__(self, oie_data_dir, IDF_path, csv_path, SUBWORDUNIT, sp_size,
                 number_of_clusters=50,
                 stemming=False,
                 stopwords=True,
                 SUBWORD_UNIT_COMBINATION="avg",
                 path_to_embeddings=None):
        self.oie_data_dir = oie_data_dir
        self.csv_path = csv_path
        self.number_of_clusters = number_of_clusters
        self.SUBWORD_UNIT_COMBINATION = SUBWORD_UNIT_COMBINATION
        self.stemming = stemming
        self.stopwords = stopwords

        self.IDF_path = IDF_path
        self.filter_data_path = IDF_path.rsplit('/', maxsplit=1)[0] + '/'

        if SUBWORDUNIT:
            self.sp_size = str(sp_size)
            self.subwordunit = SUBWORDUNIT
        else:
            self.sp_size = ''

        self.ELMo_options_path = ""
        self.ELMo_weights_path = ""
        if not path_to_embeddings:
            self.path_to_embeddings = "SORE/data/filter_data/elmo_pubmed/"
            self.check_for_embeddings()


    def check_for_embeddings(self):
        """
        May need to download the ELMo PubMed embeddings
        """
        types = ['*.hdf5', '*.json']
        embedding_files = []
        for file_type in types:
            embedding_files.extend(glob.glob(self.path_to_embeddings + file_type))

        if embedding_files == []:
            print('No embedding files found, beginning download of ELMo PubMed files.')
            w = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5"
            o = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            wget.download(w, self.path_to_embeddings + 'ELmo_PubMed_weights.hdf5')
            wget.download(o, self.path_to_embeddings + 'ELmo_PubMed_options.json')
            self.ELMo_weights_path = self.path_to_embeddings + 'ELmo_PubMed_weights.hdf5'
            self.ELMo_options_path = self.path_to_embeddings + 'ELmo_PubMed_options.json'
        elif self.path_to_embeddings +'ELmo_PubMed_weights.hdf5' in embedding_files:
            self.ELMo_weights_path = self.path_to_embeddings + 'ELmo_PubMed_weights.hdf5'
            self.ELMo_options_path = self.path_to_embeddings + 'ELmo_PubMed_options.json'
            print("Found ELMo PubMed embeddings")
            pass
        else:
            print("Assuming the ELMo PubMed embeddings are correctly set in {}".format(self.path_to_embeddings))
            # would have to add other types of embeddings


    def prepare_narrowIE_embeddings(self, prefix, sp_model_path):
        """
        If setup doesn't exist yet, runs the embedder following user-defined settings
        and stores the output in pickle files. Otherwise, loads the existing pickle files
        for the specified settings.
        """

        settings = "{pr}_{sp}{w}_{stem}_{stop}".format(pr=prefix,
                                                       sp=self.sp_size + '_',
                                                       w=str(self.SUBWORD_UNIT_COMBINATION),
                                                       stem=str(self.stemming),
                                                       stop=str(self.stopwords))

        if not os.path.exists(self.filter_data_path + "vectors/vectors_{settings}.pkl".format(settings=settings)):
            try:
                embedder = fu.PrepareEmbeddings(prefix, sp_model_path,
                                                self.sp_size,
                                                self.IDF_path,
                                                self.csv_path,
                                                self.ELMo_options_path,
                                                self.ELMo_weights_path,
                                                SUBWORD_UNIT_COMBINATION=self.SUBWORD_UNIT_COMBINATION,
                                                subwordunits=self.subwordunit,
                                                stemming=self.stemming,
                                                stopwords=self.stopwords)
                narrowIE_data = embedder.load_narrowIE_data()
                narrowIE_embeddings = embedder.embed_all_narrowIE_phrases(narrowIE_data)
            except TypeError:
                print("Narrow IE arguments not properly embedded.")
                return

            with open(self.filter_data_path + "vectors/nIE_phrases_{settings}.pkl".format(settings=settings),
                      'wb') as f:
                pickle.dump(narrowIE_data, f)

            with open(self.filter_data_path + "vectors/nIE_emb_{settings}.pkl".format(settings=settings),
                      'wb') as f:
                pickle.dump(narrowIE_embeddings, f)
        else:
            with open(self.filter_data_path + "vectors/nIE_phrases_{settings}.pkl".format(settings=settings),
                      'rb') as f:
                narrowIE_data = pickle.load(f)
            with open(self.filter_data_path + "vectors/nIE_emb_{settings}.pkl".format(settings=settings),
                      'rb') as f:
                narrowIE_embeddings = pickle.load(f)

        return narrowIE_data, narrowIE_embeddings, embedder


    def get_docid_from_filename(self, filename, output_name=False):
        if output_name:
            return self.oie_data_dir+'processed/'+filename.rsplit('/',maxsplit=1)[1][:-4]+'_processed.txt'
        return filename.rsplit('/',maxsplit=1)[1][:-4]


    def OIE_files_to_filter(self):
        """
        Tries to ensure that only the necessary 'processed OIE files' are selected,
          based on the defined narrow IE output csv_file.
        """
        input_files = glob.glob(self.oie_data_dir + '*.txt')
        doc_ids_for_filtering = []
        with open(self.csv_path, 'r') as csv_f:
            reader = csv.DictReader(csv_f)
            for row in reader:
                doc_ids_for_filtering.append(row['doc_id'])

        doc_ids_for_filtering = list(set(doc_ids_for_filtering))

        return [f for f in input_files if self.get_docid_from_filename(f) in doc_ids_for_filtering]


    def run(self, prefix, filter_settings, output_dir,
            print_clusters=False,
            plot=False,
            cluster_names=None,
            print_stats=False):

        sp_model_path = self.filter_data_path + "{}_{}.model".format(prefix, self.sp_size)

        narrowIE_phrases, narrowIE_embeddings, embedder = self.prepare_narrowIE_embeddings(prefix, sp_model_path)

        clusterer = fu.ClusterTradeOffs(self.filter_data_path, self.number_of_clusters,
                                        self.sp_size, self.stemming, self.stopwords)
        km_model = clusterer.get_Kmeans_model(narrowIE_phrases, narrowIE_embeddings)

        # need to pass the model to filter
        filterer = fu.SoreFilter(self.oie_data_dir, self.csv_path, self.IDF_path,
                                 self.subwordunit, sp_model_path,
                                 self.ELMo_weights_path, self.ELMo_options_path, filter_settings)

        filterer.start_filtering(output_dir, prefix, narrowIE_phrases, narrowIE_embeddings,
                                 embedder, km_model, print_stats)

        ## To gain some insight into the created clusters:
        if print_clusters:
            clusters, results = clusterer.cluster(km_model, narrowIE_phrases, narrowIE_embeddings)
            clusterer.cluster_insight(results)

        if plot:
            if cluster_names:
                # You can manually label the clusters if you like
                category_list = [x for x in cluster_names.values()]
            else:
                category_list = [x for x in range(self.number_of_clusters)]

            digits_proj = TSNE(random_state=self.randomstate).fit_transform(clusters)
            clusterer.palplot(digits_proj, km_model, category_list)




