# ArticleJournalClassifier
Predicting the publication venue of academic papers using a few select NLP techniques!

## Running the Code
Run all of the provided commands in the root of the project, in the same directory as the requirements.txt 
file and the README.md file, to avoid issues with file locations.

Given that this project is structured as a module, all files needs to be run with `python -m <module-path>`

### Packages Setup
This project uses several external packages and libraries, some of which you may not have globally or locally installed.

Install then from the requirements file using: `pip install -r requirements.txt`

### Preprocessing
The preprocessing directory contains files which are responsible for preprocessing the arXiv data, into usable training data.

`ArticleJournalClassifier/Preprocessing/metadata_preprocessing.py` processes the metadata file, and exports a JSON called `journal_id_map.json` 
containing journals mapped to the ids of their papers. 

To run this file, download the following from Google Drive, and place it in the root of the repo: https://drive.google.com/file/d/1c2ugbEASkO3SHEs4yzGKIV4R1rPx4cgS/view?usp=sharing
Then, run: `python -m ArticleJournalClassifier.Preprocessing.metadata_preprocessing`


`ArticleJournalClassifier/Preprocessing/pdf_processing.py` processes the journal mapping, queries for and scrapes the pdfs for each article, and exports a JSON called `top_journals_text.json`
which contains the training data in the form of articles for each of the final classes. 

To run this file, download the following from Google Drive, and place it in the root of the repo: https://drive.google.com/file/d/1CxnZ0AvjiJ-nTG7NoQWUHh94SfZryd4D/view?usp=sharing
Alternatively, you can directly use the output of the previous program, but it will take a very long time (8 hr +)
Then, run: `python -m ArticleJournalClassifier.Preprocessing.pdf_processing`

### Naive Bayes
The Naive Bayes directory contains files which are responsible for training and evaluating the Naive Bayes Model.

`ArticleJournalClassifier/Naive_Bayes/nb_classifier_training.py` trains the model, using data from `top_journals_text.json`.
The final model is exported as `nb_data/trained_nb_model.joblib`

To run this file, download the training data from Google Drive, and place it in the root of the repo: https://drive.google.com/file/d/1s7jnxfvFWHFvtOJMzW8YKcABlr77G2sS/view?usp=sharing
Then, run: `python -m ArticleJournalClassifier.Naive_Bayes.nb_classifier_training`

`ArticleJournalClassifier/Naive_Bayes/nb_classifier_eval.py` tests the model, using data from `top_journals_text.json` and the trained weights.

To run this file, download the trained weights dir from Google Drive, unzip, and place it in the root of the repo: https://drive.google.com/drive/folders/1JfWah2ARvUJifAzxwAIEKZ8Kv_QVIBwG?usp=sharing
Ensure the the training data `top_journals_text.json` has already been downloaded. 
Then, run: `python -m ArticleJournalClassifier.Naive_Bayes.nb_classifier_eval`

### RNN
The RNN directory contains the files which are responsible for training and evaluating the RNN + LSTM Model.

`ArticleJournalClassifier/RNN/rnn_train.py` trains the model, using data from `top_journals_text.json`.
The final model is exported within the directory `rnn_data`, containing the model weights, and the vocab mappings. 

To run this file, download the training data from Google Drive, and place it in the root of the repo: https://drive.google.com/file/d/1s7jnxfvFWHFvtOJMzW8YKcABlr77G2sS/view?usp=sharing
Note that if you have already downloaded it, this is the same training data.
Then, run `python -m ArticleJournalClassifier.RNN.rnn_train`

`ArticleJournalClassifier/RNN/rnn_eval.py` evaluates the model, using data from `top_journals_text.json` and the trained weights.

To run this file, download the trained weights dir from Google Drive, unzip it, and place it in the root of the repo: https://drive.google.com/drive/folders/1qDAY4tajtvJflmeSC17wDrwYrVVLYz85?usp=sharing
Then, run `python -m ArticleJournalClassifier.RNN.rnn_eval`


### BERT

The BERT directory contains the files which are responsible for fine-tuning and evaluating BERT.

`ArticleJournalClassifier/BERT/bert_training.py` fine tunes the mode, using data from `top_journals_text.json`.
The final model is exported within the directory `bert-finetuned-top-journals`, containing weights, vocab, checkpoints, etc.

To run this file, download the training data from Google Drive, and place it in the root of the repo: https://drive.google.com/file/d/1s7jnxfvFWHFvtOJMzW8YKcABlr77G2sS/view?usp=sharing
Note that if you have already downloaded it, this is the same training data.
Then, run `python -m ArticleJournalClassifier.BERT.bert_training`

`ArticleJournalClassifier/BERT/bert_eval.py` evaluates the model, using data from `top_journals_text.json`, and the weights in `bert-finetuned-top-journals`.

To run this file, downloaded the trained weights from Google Drive, unzip it, and place it in the root of the repo: https://drive.google.com/drive/folders/1uzU3ZoivxCtLYfNL32MnUIw29IoWQXYd?usp=sharing
Then, run `python -m ArticleJournalClassifier.BERT.bert_eval`


### Data

This directory contains a variety of utility functions, which are used in data preprocessing.
This code isn't independently runnable, and is already utilized within the previous files. 
