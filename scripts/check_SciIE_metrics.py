import json

input_gold = "../data/dev_set_SCIIE.json"
input_pred = "predictions_dev.json"

input_gold2 = "../data/test_set_SCIIE.json"
input_pred2 = "predictions_test.json"

class Scorer(object):
    def __init__(self, metric):
        self.precision_numerator = 0
        self.precision_denominator = 0
        self.recall_numerator = 0
        self.recall_denominator = 0
        self.metric = metric
        self.num_labels = 0


    def update(self, gold, predicted):
        p_num, p_den, r_num, r_den = self.metric(self, gold, predicted)
        self.precision_numerator += p_num
        self.precision_denominator += p_den
        self.recall_numerator += r_num
        self.recall_denominator += r_den


    def get_f1(self):
        precision = 0 if self.precision_denominator == 0 else \
            self.precision_numerator / float(self.precision_denominator)
        recall = 0 if self.recall_denominator == 0 else \
            self.recall_numerator / float(self.recall_denominator)
        return 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


    def get_recall(self):
        if self.recall_numerator == 0:
            return 0
        else:
            return self.recall_numerator / float(self.recall_denominator)


    def get_precision(self):
        if self.precision_numerator == 0:
            return 0
        else:
            return self.precision_numerator / float(self.precision_denominator)


    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1(), self.num_labels


    def f1_score(self, gold_labels, pred_labels):
        """
        ◮ Precision: (# spans correctly assigned ~ p_num) TP / (# spans assigned ~ p_den) TP+FP
        ◮ Recall: (# spans correctly assigned ~ r_num) TP / (total # of spans ~ r_den) TP+FN
        """
        p_num, p_den, r_num = 0, 0, 0
        # every gold_span that should be 1 (TP + FN)
        r_den = len(gold_labels)
        self.num_labels += r_den

        for span in pred_labels:
            p_den += 1 # all predicted spans, both true and false (TP+FP)
            if span in gold_labels: # all spans and tuples are unique (enforced during reformatting of dataset)
                p_num += 1
                r_num += 1

        return p_num, p_den, r_num, r_den


def compare_against_gold(gold_input, pred_input):

    gold_annotations = []
    with open(gold_input) as o:
        for line in o.read().split('\n'):
            if len(line) > 10:
                gold_annotations.append(json.loads(line))

    predicted_annotations = []
    with open(pred_input) as o:
        for line in o.read().split('\n'):
            if len(line) > 10:
                predicted_annotations.append(json.loads(line))

    span_scorer = Scorer(Scorer.f1_score)
    triple_scorer = Scorer(Scorer.f1_score)

    for g_ann, p_ann in zip(gold_annotations, predicted_annotations):
        if g_ann['doc_key'] != p_ann['doc_key']:
            print('Doc_key mismatch. Are you using right combination of gold annotations and predictions?')
            break
        else:
            # [0] since only one sentence
            span_scorer.update(g_ann['ner'][0], p_ann['ner'][0])
            triple_scorer.update(g_ann['relations'][0], p_ann['relation'][0])

    span_p, span_r, span_f1, total_spans = span_scorer.get_prf()
    triple_p, triple_r, triple_f1, total_tuples = triple_scorer.get_prf()

    print("Predictions for {}".format(gold_input))
    print("\t\t Prec.\t Rec.\t F1 \t #total")
    print("Spans: \t {:.2f} \t{:.2f}\t{:.2f}\t{}".format(span_p*100, span_r*100, span_f1*100, total_spans))
    print("Rel's: \t {:.2f} \t{:.2f}\t{:.2f}\t{}".format(triple_p*100, triple_r*100, triple_f1*100, total_tuples))


compare_against_gold(input_gold, input_pred)
compare_against_gold(input_gold2, input_pred2)