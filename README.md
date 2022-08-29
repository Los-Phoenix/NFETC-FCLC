### Dependencies
- python 3.6.0 +
- Other lib dependencies are listed in requirements, install them with `pip install -r requirements.txt`. 

### 1. Prepare

Download data in `data` directory using `download.sh`.
Run command:
`sed -i "1i 2196017 300" data/glove.840B.300d.txt` to insert a line "2196017 300" to the head of `data/glove.840.300d.txt`.

### 2. Preprocess

To generate `wikim`(wiki with the improved hierarchy) raw data, run `python transform_wiki.py` first.

To generate the Wikim/Ontonotes dataset, run `python preprocess.py -d <data_name>` and fill `<data_name>` with `wikim` or `ontonotes`.

We use the preprocessed `BBN`dataset [released by NFETC-CLSC](https://drive.google.com/open?id=1opjfoA0I2mOjE11kM_TYsaeq-HHO1rqv). 


### 3. Train and Evaluation

Run `python eval.py -m <model_name> -d <data_name> -r <runs> -a <alpha>` and the scores for each run and the average scores are recorded in one log file stored in folder `log`.

- `<data_name>` choices: `wikim, ontonotes, bbn`
- `<model_name>` choices: `wikim, ontonotes, bbn`
- `<alpha>` is the hierarchy loss factor.

To re-implement the results on the three datasets, run following commands:

- Wikim: `python eval4test.py -m wikim -d wikim -r 5 -a 0.4`
- OntoNotes: `python eval4test.py -m ontonotes -d ontonotes -r 5 -a 0.2`
- BBN: `python eval4test.py -m bbn -d bbn -r 5 -a 0`

(Sorry for the delay, the author was overwhelmed by geting the degree and graduating. Plz feel free to start an issue if you have reproduction problems such as not running or not as high scores since ome hyper-paramerters may not be the optmistic setup in the paper. In emergency cases, plz email pangkunyuan@hotmail.com)