
Install requirements:
  * `pip install -r requirements.txt`
  * Download required language models
    * `python -m spacy download en_core_web_sm`
    * `python -m textblob.download_corpora`


----
### Run SciIE
Clone the SciIE code from [this bitbucket repository](https://bitbucket.org/luanyi/scierc/src/master/)

Copy the trained model 
Create a SCIIE environment with: 
  * Python 2.7
  * TensorFlow 1.8.0 
  * tensorflow_hub
  * pyhocon

Download a model trained on FOBIE [here](https://), and place it inside the SciIE folder under the `logs/` directory.

##### 1) Preparation
Open the SciIE folder in a terminal/console window and run:
  * `./scripts/fetch_required_data.sh`
  * `./scripts/build_custom_kernels.sh`

##### 2) Generate elmo embeddings
  * Copy the generated input file  `narrowIE/example_input.json` to the SciIE data folder
  * Make sure to change the input and output for `generate_elmo.py`:
    * fn = './data/processed_data/json/example_input.json'
    * outfn = './data/processed_data/elmo/example_input.hdf5'

Then run: `python generate_elmo.py ` (make sure to be in the correct environment)

##### 3) Predict using trained model on example_input
Open experiments.conf and change your experimental setup:
  * lm_path_dev = "./data/processed_data/elmo/example_input.hdf5"
  * eval_path = "./data/processed_data/json/example_input.json"
  * output_path = "./data/predictions_example_input.json"

Then run write_single.py on your experimental setup (e.g. scientific_elmo):  
` python write_single.py scientific_elmo `

<!-- Note: if training your own model, you have to stop running the training session manually. We followed patience=12.  -->

----
### Prepare OpenIE 5
Clone the [github repo for OpenIE 5](https://github.com/dair-iitd/OpenIE-standalone)

Open the newly created `OpenIE-standalone` and create a directory called `lib/`.  
You will have to download and place into the `lib/` folder:
  * [The BONIE standalone jar](https://github.com/dair-iitd/OpenIE-standalone/releases/download/v5.0/BONIE.jar)
  * [The CALMIE standalone jar](https://github.com/dair-iitd/OpenIE-standalone/releases/download/v5.0/ListExtractor.jar)

Furthermore, you will have to download and place into the `data/` folder: 
  * [The Berkeley Language Model](https://drive.google.com/file/d/0B-5EkZMOlIt2cFdjYUJZdGxSREU/view?usp=sharing)


OpenIE is compiled using `sbt` and Java 8:  
  * Install [SDK](https://sdkman.io/install) by running `curl -s "https://get.sdkman.io" | bash`  
  * Open a new terminal window and check that sdk is installed `sdk version`
  * Then install sbt `sdk install sbt`:
    * You may need sbt at version 0.13.x , e.g.: `sdk install sbt 0.13.18`
  * Inside the `OpenIE-standalone/` directory:
    * First run  `bash compile.sh` to compile SRLIE and ONRE
    * Then compile the jar file `sbt -J-Xmx10000M clean compile assembly`  
    * You may have to [install Java 8](https://www.scala-sbt.org/1.x/docs/Installing-sbt-on-Mac.html) first , e.g.: `sdk install java 8.0.252-amzn`
    * And add java to your path, e.g., `export PATH=$PATH:~/.sdkman/candidates/java/8.0.252-amzn/bin`
    * You may need to set the Scala version to [version 2.10.2](https://www.scala-lang.org/download/2.10.2.html)

  
Once done, be sure to change the variable `path_to_OIE_jar` in `runOIE5.py`. An example path:  
`/Users/..../dev/OpenIE-standalone/target/scala-2.10/openie-assembly-5.0-SNAPSHOT.jar`
