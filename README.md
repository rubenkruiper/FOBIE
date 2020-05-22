Semi-Open Relation Extraction
--

Computer-Aided Biomimetics dataset for the extraction of TRADE-OFFS and syntactically similar relations from scientific biological texts.

**Readme will be cleaned soon.**
**Code for Semi-Open Relation Extraction will be added soon.**

### Format
The train/test/dev data files are provided in two formats. A verbose json format inspired on the Semeval2018 task 7 dataset:

```
<sub>{"[document_ID]": </sub>
<sub>  {"[relation_ID_within_document]": </sub>
<sub>    {"annotations": </sub>
<sub>      {"modifiers": </sub>
<sub>        {"[within_sentence_modifier_ID]": </sub>
<sub>          {"Arg0": {"span_start": "[token_index]", </sub>
<sub>                    "span_end": "[token_index]", </sub>
<sub>                    "span_id": "[brat_ID]",  </sub>
<sub>                    "text": "[string]"}, </sub>
<sub>           "Arg1": {"span_start": "[token_index]", </sub>
<sub>                    "span_end": "[token_index]", </sub>
<sub>                    "span_id": "[brat_ID]", </sub>
<sub>                    "text": "[string]"}</sub>
<sub>          }</sub>
<sub>       }, </sub>
<sub>     "tradeoffs": </sub>
<sub>        {"[within_sentence_tradeoff_ID]": </sub>
<sub>          {"Arg0": {"span_start": "[token_index]", </sub>
<sub>                    "span_end": "[token_index]", </sub>
<sub>                    "span_id": "[brat_ID]",  </sub>
<sub>                    "text": "[string]"}, </sub>
<sub>          "Arg1": {"span_start": "[token_index]", </sub>
<sub>                   "span_end": "[token_index]", </sub>
<sub>                   "span_id": "[brat_ID]",  </sub>
<sub>                   "text": "[string]"},        </sub>   
<sub>          "TO_indicator": {"span_start": "[token_index]",</sub> 
<sub>                           "span_end": "[token_index]", </sub>
<sub>                           "span_id": "[brat_ID]",  </sub>
<sub>                           "text": "[string]"}, </sub>
<sub>          "labels": {"Confidence": "High"}</sub>
<sub>        }</sub>
<sub>      }</sub>
<sub>    }, </sub>
<sub>    "sentence": "[string]"</sub>
<sub>    }</sub>
<sub>}, {[next document]} </sub>
``` 

And the Sci-ERC dataset format, which is used to train the SciIE system:
```
<sub>[line1] {   "clusters": [],</sub>
<sub>              "sentences": [["List", "of", "some", "tokens", "."]],</sub>
<sub>              "ner": [[[4, 4, "Generic"]]],</sub>
<sub>              "relations": [[[4, 4, 6, 17, "Tradeoff"]]],</sub>
<sub>              "doc_key": "XXX"}</sub>
<sub>[line2] {   [next sentence]</sub>
``` 

We also provide a script to convert data from our format to SciIE format, as well as a script to convert BRAT annotations to the verbose format.

### Statistics

|<sub>                  </sub>| <sub>Train</sub>| <sub>Dev</sub>   | <sub>Test</sub>  | <sub>Total</sub> |
|-----------------------------|-------------|-------|-------|-------|
| <sub># Sentences</sub>                 | <sub>1248</sub>        | <sub>150</sub>   | <sub>150</sub>   | <sub>1548</sub>  |
| <sub>Avg. sent. length</sub>           | <sub>37.28</sub>       | <sub>37.78</sub> | <sub>37.82</sub> | <sub>37.77</sub> |
| <sub>% of sents ≥ 25 tokens</sub>      | -           | -     | -     | <sub>79.26</sub> |
| Relations:                  |             |       |       |       |
|  * <sub>Trade-Off</sub>                 | <sub>639</sub>         | <sub>54</sub>    | <sub>72</sub>    | <sub>765</sub>   |
|  * <sub>Not-a-Trade-Off</sub>           | <sub>2004</sub>        | <sub>258</sub>   | <sub>240</sub>   | <sub>2502</sub>  |
|  * <sub>Arg-Modifier</sub>              | <sub>1247</sub>        | <sub>142</sub>   | <sub>132</sub>   | <sub>1521</sub>  |
| <sub>Triggers</sub>                    | <sub>1292</sub>        | <sub>155</sub>   | <sub>153</sub>   | <sub>1600</sub>  |
| <sub>Arguments</sub>                   | <sub>3435</sub>        | <sub>401</sub>   | <sub>398</sub>   | <sub>4234</sub>  |
| <sub>Spans</sub>                       | <sub>5137</sub>        | <sub>596</sub>   | <sub>576</sub>   | <sub>6309</sub>  |
| <sub>Unique spans</sub>                | <sub>2701</sub>        
| <sub>Unique triggers</sub>             | <sub>41 </sub>         
| <sub>Max triggers/sent</sub>           | <sub>2 </sub>          
| <sub>Max spans/sent</sub>              | <sub>7</sub>           
| <sub>Spans w/ multiple relations</sub> | <sub>2075</sub>        
| <sub># single-word arguments</sub>     | <sub>498 (11.8%) </sub>
| <sub>Avg. tokens per argument</sub>    | <sub>3.44 </sub>       


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


