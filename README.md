# STFT Transformer
Code for STFT Transformer used in BirdCLEF 2021 competition.

The STFT Transformer is a new way to use Transformers similar to Vision Transformers on audio data.  It has been developed for the [BirdCLEF 2021 competition hosted on Kaggle](https://www.kaggle.com/c/birdclef-2021).  The pdf document gives more context.  It has been submitted to the BIRDCLEF 2021 workshop.

The code is provided as is, it has not been rewritten.  Given competitions are done in a hurry, code may not meet usual open source standard.

The code assumes this directory structure:

<base_dir>/code

<base_dir>/input

<base_dir>/input/freefield1010

<base_dir>/checkpoints

<base_dir>/data

Code has to be run in the code directory.  Competition data has to be downloaded in the input directory.  freefield1010 data must also be downloaded in the freefield1010 directory. data_final.py should be run first. It reads audio files from input and stores the relevant part in data directory as numpy files.

Then stft_transformer_final.py can be run to train one fold model.  During the competition I ran 5 folds, by editing the FOLD global variable in the script (I know, this is sub standard).

Once all 5 models are trained one can upload the weights to a kaggle dataset and [use the submission notebook](https://www.kaggle.com/cpmpml/stft-transformer-infer?scriptVersionId=65743541) I used.  This should get a score worth the 15th rank in the competition.  Achieving this rank with a single model is significant, as all top teams used an ensemble of models.
