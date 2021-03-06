import json, glob, re, csv
from collections import Counter
from SORE.my_utils.filter_utils import get_stats_filtered, get_stats_unfiltered


class BratConverter():
    """
    Encapsulates the paths to convert SORE filtered files to BRAT annotations (for visualisation of results).
    """
    def __init__(self, paths_to_datasets, narrowIE_path, SORE_processed_path, BRAT_output_path):
        """
        Initialise BratConverter with relevant paths.

        :param paths_to_datasets: List of paths to unprocessed data sets. Can be used to retrieve document meta-data, such as
            the category of a document in the OA-STM corpus. If these categories are found, the stats per category will be printed.
        :param narrowIE_path: Path to CSV file with narrow IE extractions.
        :param SORE_processed_path: Path to directory with the files that were filtered with SORE. Unfiltered json-files
            can be found here as well, so that the entire document can be converted to BRAT (not just filtered sentences).
        :param BRAT_output_path: Path to write the output .ann and .txt files for visualisation with BRAT
        """
        self.paths_to_datasets = paths_to_datasets
        self.narrowIE_csv = narrowIE_path
        self.SORE_path = SORE_processed_path
        self.BRAT_output_path = BRAT_output_path

    def get_complete_span(self, w_list, sentence):
        """
        Determine the string in the original sentence spanned by the first and last word of an argument-list.

        :param w_list: List of words for an argument.
        :param sentence: Sentence from which the argument was extracted.
        :return: The string in the sentence that corresponds to the argument-list.
        """
        str_to_search = None
        w_list = [w for w in w_list if w not in [")", "("]]
        if len(w_list) == 1:
            str_to_search = w_list[0]
            str_to_search = str_to_search.replace('(', "\(").replace(')', "\)")
        elif len(w_list) > 1:
            try:
                str_to_search = w_list[0] + re.search(w_list[0] + "(.*?)" + w_list[-1], sentence).group(1) + w_list[-1]
                # str_to_search = str_to_search.replace('(', "\(").replace(')', "\)")
            except:
                print("Regex issue ignored for now: {}".format(w_list))

        return str_to_search


    def prep_ann_line(self, span_type, text, sentence, sent_start_idx, span_counter):
        """
        Prepare a line that should be written to the .ann file, for a given span.

        :param text: A span of text (argument represented as string) found by SORE (or OpenIE if relevant lines are uncommented).
        :param sentence: String with the sentence from the original input, in which the span was found.
        :param sent_start_idx: Index of the sentence start in the whole document
        :param span_counter: Index of the previous span annotation in the whole document
        :return: The line to write to an .ann file, and the updated span_counter
        """
        rel_match = None
        if text and text in sentence:
            sc = span_counter + 1
            s = sentence.find(text)
            e = s + len(text) + sent_start_idx
            return "T{}\t{} {} {}\t{}".format(sc, span_type, str(s + sent_start_idx), str(e), text), sc
        elif text:
            str_to_search = text.replace('(', "\(").replace(')', "\)").replace('[', " ").replace(']', " ")
            try:
                rel_match = re.search(str_to_search, sentence)
            except:
                print("Regex issue ignored for now: {}".format(str_to_search))

        if rel_match:
            sc = span_counter + 1
            s = rel_match.start() + sent_start_idx
            e = rel_match.end() + sent_start_idx
            return "T{}\t{} {} {}\t{}".format(sc, span_type, str(s), str(e), text), sc
        else:
            return "", span_counter


    def convert_NIE_annotations(self, sentence, narrowIE_args, sent_start_idx,
                                span_counter, rel_counter):
        """
        Converts the narrow IE annotations for a sentence to a list of annotations to write to the .ann file.

        :param sentence: String with input sentence.
        :param narrowIE_args: Argument phrases found through narrow IE in that sentence.
        :param sent_start_idx: Index of the sentence start in the whole document
        :param span_counter: Index of the previous span annotation in the whole document
        :param rel_counter: Index of the previous relation annotation in the whole document
        :return: Lines to write for the narrow IE extractions in this sentence, and updated counters.
        """
        lines_to_write = []

        narrowIE_unique_args = []
        for arg in narrowIE_args[0]:
            if arg not in narrowIE_unique_args:
                narrowIE_unique_args.append(arg)


        # narrow IE spans
        NIE_span_ids = []
        for str_to_search in narrowIE_unique_args:
            NIE_arg_span, span_counter = self.prep_ann_line("NarrowIE_span", str_to_search,
                                                                sentence, sent_start_idx, span_counter)
            if NIE_arg_span != "":
                NIE_span_ids.append(span_counter)
                lines_to_write.append(NIE_arg_span)

        # NIE relations - R1	ArgMod Arg0:T3 Arg1:T1
        while len(NIE_span_ids) > 1:
            a1, a2 = NIE_span_ids[:2]
            NIE_span_ids.pop(0)
            # \t at the end seems required.
            relation = "R{}\tNIE_extraction Arg0:T{} Arg1:T{}\t".format(rel_counter, a1, a2)
            rel_counter += 1
            lines_to_write.append(relation)

        return lines_to_write, span_counter, rel_counter


    def convert_OIE_annotations(self, sentence, annotations, sent_start_idx,
                                span_counter, event_counter, attribute_counter):
        """
        Converts an Open IE  extraction (after/before filtering) for a sentence to a list of annotations to write.

        :param sentence: String with input sentence.
        :param annotations: One of possibly multiple Open IE extraction for this sentence
        :param sent_start_idx: Index of the sentence start in the whole document
        :param span_counter: Index of the previous span annotation in the whole document
        :param event_counter: Index of the previous Open IE event annotation in the whole document
        :param attribute_counter: Index of the previous event attribute in the whole document
        :return: Lines to write for one Open IE extraction in this sentence, and updated counters.
        """

        lines_to_write = []
        attributes = []

        event_span_ids = []
        # OIE relation span
        OIE_rel = annotations['rel']
        rel_span, span_counter = self.prep_ann_line("OpenIE_rel_span", OIE_rel, sentence, sent_start_idx, span_counter)
        if rel_span == "":
            # The relation may consist of two separate spans, avoid dealing with this for now
            parts = OIE_rel.split(" ")
            rel_span, span_counter = self.prep_ann_line("OpenIE_rel_span", max(parts, key=len), sentence, sent_start_idx,
                                                        span_counter)

        if rel_span != "":
            event_span_ids.append(span_counter)
            lines_to_write.append(rel_span)

        # OIE argument spans
        OIE_arg_list = []
        for arg in annotations['args']:
            w_list, _ = arg                     # recall token_length may not match up due to stopword removal
            if w_list != []:
                OIE_arg_list.append(w_list)

        for w_list in OIE_arg_list:
            str_to_search = self.get_complete_span(w_list, sentence)
            arg_span, span_counter = self.prep_ann_line("OpenIE_span", str_to_search,
                                                        sentence, sent_start_idx, span_counter)
            if arg_span != "":
                lines_to_write.append(arg_span)
                # in case an issue occured with OIE_rel_span
                if event_span_ids != []:
                    event_span_ids.append(span_counter)

        # in case no issue occurred with getting the OIE_rel_span
        if event_span_ids != []:
            # context spans
            context_span = None
            if annotations['context'] != "context()":
                str_to_search = annotations['context'][8:-1]
                context_span, span_counter = self.prep_ann_line("Context_span", str_to_search,
                                                                sentence, sent_start_idx, span_counter)
                context_id = span_counter
                if context_span != "":
                    lines_to_write.append(context_span)

            # OIE attributes
            if float(annotations['conf']) > 0.8:
                confidence = "High"
            else:
                confidence = "Low"

            attributes.append("A{}\tConfidence E{} {}".format(attribute_counter, event_counter, confidence))
            attribute_counter += 1
            if annotations['negation']:
                attributes.append("A{}\tNegation E{}".format(attribute_counter, event_counter, confidence))
                attribute_counter += 1

            # OIE event
            event = "E{}\tOpenIE_rel_span:T{} ".format(event_counter, str(event_span_ids.pop(0)))
            event_counter += 1
            for idx, arg in enumerate(event_span_ids):
                event += "Arg{}:T{} ".format(idx, arg)
            if context_span:
                event += "Context:T{}".format(context_id)

            lines_to_write.append(event)
            lines_to_write += attributes

        return lines_to_write, span_counter, event_counter, attribute_counter


    def parse_argument_list(self, string_list):
        """
        Parses a narrow IE argument from the narrow IE CSV file (duplicate of narrowIE_parser).

        :param string_list: String that holds the list of arguments in the CSV file.
        :return: List of argument-phrases
        """
        return [x[1: -1] for x in string_list[1: -1].split(", ")]


    def convert_to_BRAT(self, prefix):
        """
        Convert the SORE extractions to BRAT annotations.

        :param prefix: Experiment name prefix for SORE files to convert.
        """
        dataset = {}
        for dataset_path in self.paths_to_datasets:
            with open(dataset_path) as d:
                dataset.update(json.load(d))

        OIE_dict = {}
        SORE_dict = {}

        unfiltered_files = glob.glob(self.SORE_path + prefix +"*_unfiltered.json")
        filtered_files = glob.glob(self.SORE_path + prefix +"*[0-9].json")

        for annotation_file in unfiltered_files:
            with open(annotation_file) as f:
                OIE_dict.update(json.load(f))

        for annotation_file in filtered_files:
            with open(annotation_file) as f:
                SORE_dict.update(json.load(f))

        # retrieve the sentences and sent_ids for the narrow IE extractions
        narrowIE_dict = {}
        with open(self.narrowIE_csv, 'r') as csv_f:
            reader = csv.DictReader(csv_f)
            for row in reader:
                doc_id = row['doc_id']
                sent_id = row['sentence_nr']
                relation_types = row['relation_types']
                argument_list = list(self.parse_argument_list(row['arguments']))

                if doc_id in narrowIE_dict:
                    narrowIE_dict[doc_id].update({sent_id: [argument_list],
                                                  "rel_types": relation_types})
                else:
                    narrowIE_dict[doc_id] = {sent_id: [argument_list],
                                             "rel_types": relation_types}


        # because the OA-STM dataset has categories, use these to get some insight into filtering results per category
        OIE_dicts_per_category = {}
        SORE_dicts_per_category = {}

        for doc_id in SORE_dict.keys():

            try:
                if dataset[doc_id]['metadata'] != None:
                    category = dataset[doc_id]['metadata']['category']
                else:
                    category = 'No_categories'
            except KeyError:
                # This occurs when the original doc_id contained '.', which is replaced for '_' somewhere
                category = 'No_categories'


            if category in OIE_dicts_per_category:
                OIE_dicts_per_category[category].update({doc_id: OIE_dict[doc_id]})
                SORE_dicts_per_category[category].update({doc_id: SORE_dict[doc_id]})
            else:
                OIE_dicts_per_category[category] = {doc_id: OIE_dict[doc_id]}
                SORE_dicts_per_category[category] = {doc_id: SORE_dict[doc_id]}

            # Convert SORE spans back to BRAT annotations againto
            # if category in ["Computer Science", "Biology"]: #

            # need a .txt file and .ann file per doc_id
            ann_file = self.BRAT_output_path + prefix + '[' + doc_id + '].ann'
            txt_file = self.BRAT_output_path + prefix + '[' + doc_id + '].txt'

            # Collect all sentences, and all (filtered) annotations for a document
            all_sentences = ""
            lines_for_document = []
            sent_start_idx = 0

            doc_span_cnt = 0
            doc_attr_cnt = 1
            doc_event_cnt = 1
            doc_rel_cnt = 1

            for sent_id, sent_ in OIE_dict[doc_id].items():
                sentence = sent_[0].replace('(', "\(").replace(')', "\)")

                lines_for_sent = []
                ### Narrow IE extractions
                if sent_id in narrowIE_dict[doc_id]:
                    narrowIE_args = narrowIE_dict[doc_id][sent_id]
                    anns, doc_span_cnt, doc_rel_cnt = self.convert_NIE_annotations(sentence, narrowIE_args,
                                                                                     sent_start_idx, doc_span_cnt,
                                                                                     doc_rel_cnt)
                    lines_for_sent += anns

                # ### SORE extractions
                if sent_id in SORE_dict[doc_id]:
                    extractions = SORE_dict[doc_id][sent_id][1:]
                    for extraction in extractions:
                        anns, doc_span_cnt, doc_event_cnt, doc_attr_cnt = self.convert_OIE_annotations(sentence,
                                                                    extraction['extraction'], sent_start_idx,
                                                                    doc_span_cnt, doc_attr_cnt, doc_event_cnt)

                        # Might be able to handle redundancy here, or earlier on
                        lines_for_sent += anns # [ann for ann in anns if ann not in lines_for_sent]

                sent_start_idx += len(sentence)
                all_sentences += sentence
                lines_for_document += lines_for_sent

                ############## only here to visualise the insane amount of OIE extractions ###
                ## uncomment the code below, and comment the `Narrow IE extractions` and `SORE extractions` loops
                # if sent_id in OIE_dict[doc_id]:
                #     extractions = OIE_dict[doc_id][sent_id][1:]
                #     for extraction in extractions:
                #         anns, doc_span_cnt, doc_event_cnt, doc_attr_cnt = self.convert_OIE_annotations(sentence,
                #                                                                                        extraction,
                #                                                                                        sent_start_idx,
                #                                                                                        doc_span_cnt,
                #                                                                                        doc_attr_cnt,
                #                                                                                        doc_event_cnt)
                #
                #         # Might be able to handle redundancy here, or earlier on
                #         lines_for_sent += anns  # [ann for ann in anns if ann not in lines_for_sent]
                #
                # sent_start_idx += len(sentence)
                # all_sentences += sentence
                # lines_for_document += lines_for_sent
                ############## only here to visualise ALL OIE extractions ###

            with open(ann_file, 'w') as f:
                for line in lines_for_document:
                    if line.rstrip() != "":
                        f.writelines(line + "\n")

            with open(txt_file, 'w') as f:
                f.write(all_sentences)

        for category in OIE_dicts_per_category.keys():
            print("\nCATEGORY:  {}".format(category))
            get_stats_unfiltered(OIE_dicts_per_category[category])
            get_stats_filtered(SORE_dicts_per_category[category])