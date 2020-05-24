Semi-Open Relation Extraction
--

Computer-Aided Biomimetics dataset for the extraction of TRADE-OFFS and syntactically similar relations from scientific biological texts.

**Code for Semi-Open Relation Extraction will be added soon.**

### Format
The train/test/dev data files are provided in two formats. A verbose json format inspired on the Semeval2018 task 7 dataset:

```json
{"[document_ID]":
  {"[relation_ID_within_document]":
    {"annotations":
      {"modifiers":
        {"[within_sentence_modifier_ID]":
          {"Arg0": {"span_start": "[token_index]",
                    "span_end": "[token_index]",
                    "span_id": "[brat_ID]",
                    "text": "[string]"},
           "Arg1": {"span_start": "[token_index]",
                    "span_end": "[token_index]",
                    "span_id": "[brat_ID]",
                    "text": "[string]"}
          }
       },
     "tradeoffs":
        {"[within_sentence_tradeoff_ID]":
          {"Arg0": {"span_start": "[token_index]",
                    "span_end": "[token_index]",
                    "span_id": "[brat_ID]",  
                    "text": "[string]"},
          "Arg1": {"span_start": "[token_index]",
                   "span_end": "[token_index]",
                   "span_id": "[brat_ID]",  
                   "text": "[string]"},           
          "TO_indicator": {"span_start": "[token_index]",
                           "span_end": "[token_index]",
                           "span_id": "[brat_ID]",  
                           "text": "[string]"},
          "labels": {"Confidence": "High"}
        }
      }
    },
    "sentence": "[string]"
  }
},
```

And the Sci-ERC dataset format, which is used to train the SciIE system:
```json
{   "clusters": [],
    "sentences": [["List", "of", "some", "tokens", "."]],
    "ner": [[[4, 4, "Generic"]]],
    "relations": [[[4, 4, 6, 17, "Tradeoff"]]],
    "doc_key": "XXX"}
```

We also provide a script to convert data from the verbose format to SciIE format, as well as a script to convert BRAT annotations to the verbose format.

### Statistics
Also see _dataset_statistics.py_ under the scripts folder.
|                             |  Train | Dev   | Test  | Total |
|-----------------------------|-------------|-------|-------|-------|
| <sub># Unique documents </sub>         | <sub>1010</sub>        | <sub>138</sub>   | <sub>144</sub>   | <sub>1292</sub>  |
| <sub># Sentences</sub>                 | <sub>1248</sub>        | <sub>150</sub>   | <sub>150</sub>   | <sub>1548</sub>  |
| <sub>Avg. sent. length</sub>           | <sub>37.42</sub>       | <sub>38.91</sub> | <sub>40.02</sub> | <sub>37.81</sub> |
| <sub>% of sents ≥ 25 tokens</sub>      | <sub>82.21 %</sub>       | <sub>85.33 %</sub> | <sub>83.33 %</sub> | <sub>82.62%</sub> |
| <sub>Relations:</sub>                  |             |       |       |       |
|<sub> - Trade-Off</sub>                 | <sub>639</sub>         | <sub>54</sub>    | <sub>72</sub>    | <sub>765</sub>   |
|<sub> - Not-a-Trade-Off</sub>           | <sub>2004</sub>        | <sub>258</sub>   | <sub>240</sub>   | <sub>2502</sub>  |
|<sub> - Arg-Modifier</sub>              | <sub>1296</sub>        | <sub>147</sub>   | <sub>137</sub>   | <sub>1580</sub>  |
| <sub>Triggers</sub>                    | <sub>1292</sub>        | <sub>155</sub>   | <sub>153</sub>   | <sub>1600</sub>  |
| <sub>Keyphrases</sub>                   | <sub>3451</sub>        | <sub>401</sub>   | <sub>400</sub>   | <sub>4252</sub>  |
| <sub>Keyphrases w/ multiple relations</sub> | <sub>1614</sub>        | <sub>189</sub>   | <sub>164</sub>   | <sub>1967</sub> |
| <sub>Spans</sub>                       | <sub>4743</sub>        | <sub>556</sub>   | <sub>553</sub>   | <sub>5852</sub>  |
| <sub>Max triggers/sent</sub>           | <sub>2 </sub> | <sub>2 </sub> | <sub>2 </sub> |         
| <sub>Max spans/sent</sub>              | <sub>11</sub>  | <sub>8 </sub> | <sub>10 </sub> |
| <sub>Unique spans</sub>                | |       |       |<sub>3643</sub>   |      
| <sub>Unique triggers</sub>             | |       |       |<sub>41 </sub>    |             
| <sub># single-word keyphrases</sub>     | |       |       |<sub>872 (20.51%) </sub>|
| <sub>Avg. tokens per keyphrase</sub>    | |       |       |<sub>3.45 </sub>     |   


If you use the FOBIE dataset in your research, please cite the following paper:
```
@inproceedings{Kruiper2020_FOBIE,
  author =      "Kruiper, Ruben
                and Vincent, Julian F V
                and Chen-Burger, Jessica
                and Desmulliez, Marc P Y
                and Konstas, Ioannis",
  title =       "A Scientific Information Extraction Dataset for Nature Inspired Engineering"
  booktitle =   "Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)",
  year =        "2020",
  keywords =    "Biomimetics,Relation Extraction,Scientific Information Extraction,Trade-Offs",
  pages =       "2078--2085",
  url =         "http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.255.pdf",
  arxivId =     "2005.07753v1"
}
```



The FOBIE dataset along with scripts in this repository are licensed under a Creative Commons Attribution 4.0 License.
<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-sa.png" width="134" height="47">
