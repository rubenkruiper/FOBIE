FOBIE
--
Computer-Aided Biomimetics dataset for the extraction of TRADE-OFFS and syntactically similar relations from scientific biological texts.

**Code for Semi-Open Relation Extraction will be added soon.**
<!-- Dataset: json & SciIE format, Script: rewrite this, Link to paper: once it's on arxiv, Bibtex: https://github.com/multi30k/dataset -->

### Format
The train/test/dev data files are provided in two formats. A verbose json format inspired on the Semeval2018 task 7 dataset:
```
{"[document_ID]": 
  {"[relation_ID_within_document]": 
    {"annotations": 
      {"modifiers": 
        {"[within_sentence_modifier_ID]": 
          {"Arg0": {"span_start": "[token_index]", "span_end": "[token_index]", "span_id": "[brat_ID]",  "text": "[string]"}, 
           "Arg1": {"span_start": "[token_index]", "span_end": "[token_index]", "span_id": "[brat_ID]", "text": "[string]"}
          }
       }, 
     "tradeoffs": 
        {"[within_sentence_tradeoff_ID]": 
          {"Arg0": {"span_start": "[token_index]", "span_end": "[token_index]", "span_id": "[brat_ID]",  "text": "[string]"}, 
          "Arg1": {"span_start": "[token_index]", "span_end": "[token_index]", "span_id": "[brat_ID]",  "text": "[string]"},           "TO_indicator": {"span_start": "[token_index]", "span_end": "[token_index]", "span_id": "[brat_ID]",  "text": "[string]"}, 
          "labels": {"Confidence": "High"}
        }
      }
    }, 
    "sentence": "[string]"}},
```

And the Sci-ERC dataset format, which is used to train the SciIE system:
```
line1 {   "clusters": [],
              "sentences": [["List", "of", "some", "tokens", "."]],
              "ner": [[[4, 4, "Generic"]]],
              "relations": [[[4, 4, 6, 17, "Tradeoff"]]],
              "doc_key": "XXX"}
line2 {   " ...
```

We also provide a script to convert data from our format to SciIE format, as well as a script to convert BRAT annotations to the verbose format.

### Statistics
|                             | Train       | Dev   | Test  | Total |
|-----------------------------|-------------|-------|-------|-------|
| # Sentences                 | 1248        | 150   | 150   | 1548  |
| Avg. sent. length           | 37.28       | 37.78 | 37.82 | 37.77 |
| % of sents ≥ 25 tokens      | -           | -     | -     | 79.26 |
|-----------------------------|-------------|-------|-------|-------|
| Relations:                  |             |       |       |       |
| - Trade-Off                 | 639         | 54    | 72    | 765   |
| - Not-a-Trade-Off           | 2004        | 258   | 240   | 2502  |
| - Arg-Modifier              | 1247        | 142   | 132   | 1521  |
|-----------------------------|-------------|-------|-------|-------|
| Triggers                    | 1292        | 155   | 153   | 1600  |
| Arguments                   | 3435        | 401   | 398   | 4234  |
| Spans                       | 5137        | 596   | 576   | 6309  |
| Unique spans                | 2701        
| Unique triggers             | 41          
| Max triggers/sent           | 2           
| Max spans/sent              | 7           
| Spans w/ multiple relations | 2075        
| # single-word arguments     | 498 (11.8%) 
| Avg. tokens per argument    | 3.44        


If you use the FOBIE dataset in your research, please cite the following paper:
```
@inproceedings{Kruiper2020_FOBIE,
address = {Marseille},
arxivId = {2005.07753v1},
author = {Kruiper, Ruben and Vincent, Julian F V and Chen-Burger, Jessica and Desmulliez, Marc P Y and Konstas, Ioannis},
booktitle = {Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)},
keywords = {Biomimetics,Relation Extraction,Scientific Information Extraction,Trade-Offs},
pages = {2078--2085},
title = {{A Scientific Information Extraction Dataset for Nature Inspired Engineering}},
url = {http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.255.pdf},
year = {2020}
}
```



The FOBIE dataset along with scripts in this repository are licensed under a Creative Commons Attribution 4.0 License.
<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-sa.png" width="134" height="47">


