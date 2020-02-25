# kaggle-nlp
Repo containing all code for the Kaggle competition "Real or Not? NLP with Disaster Tweets".

## Set Up

When cloning the repo, you must:

- Create a new folder (that will be ignored) named `data`
- Create all subfolders to match the following structure inside `data`. 
- Put all the downloaded files to `data/0_raw`.

```
/kaggle-nlp
    /main.py
    /data
        /0_raw: raw data from Kaggle
        /1_processed: clean data
        /2_split: data partitioned in train/test, CV, etc.
        /3_prediction: predictions for the split data.
        /4_submission: predictions ready to be submitted.
    /model: models used for prediction.
    /tasks
        /etl: from raw to processed
        /split: from processed to split
        /prediction: from split to prediction, and saves model in /model
        /submission: from prediction to submission
    /utils
    /settings
```
