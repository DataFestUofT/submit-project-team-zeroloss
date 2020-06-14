# Team ZeroLoss 
ASA DataFest COVID-19 Virtual Challenge @UofT

## Team Members
| Member Name  	| U of T Email                  	|
|--------------	|-------------------------------	|
| Yuchen Wang  	| raina.wang@mail.utoronto.ca   	|
| Tingfeng Xia 	| tingfeng.xia@mail.utoronto.ca 	|
| Gongyi Shi   	| gongyi.shi@mail.utoronto.ca   	|
| Chen Ding    	| chen.ding@mail.utoronto.ca    	|

## [Click Here For Submission Details](./team_submission.md)

## Tree Map of this repo
``````
├── README.md
├── cluster
│   ├── experiments
│   │   └── unprocessed_bs64_lr3.0e-02_drop0_nlayer3_hiddim256
│   │       ├── bs64_lr3.0e-02_drop0_nlayer3_hiddim256.out
│   │       └── bs64_lr3.0e-02_drop0_nlayer3_hiddim256.png
│   ├── gen_sbatch.py
│   └── model.py 
├── data
│   ├── hydrated_data
│   │   ├── api
│   │   │   ├── Makefile
│   │   │   ├── api_keys.json
│   │   │   └── get_metadata.py
│   │   └── intermediate
│   │       ├── emoji_dictionary.csv
│   │       └── us_codes.csv
│   ├── process_scripts
│   │   ├── clean_location.R
│   │   ├── clean_processed.csv_tweets.ipynb
│   │   ├── make_word_cloud.ipynb
│   │   ├── process_json.R
│   │   ├── process_train.R
│   │   ├── replace_emoji.ipynb
│   │   ├── sample_ids.R
│   │   ├── sentiment_visual.R
│   │   └── text_processing.ipynb
│   ├── training_data
│   │   └── final_train_data.csv
│   └── word_cloud_data
│       ├── negative.csv
│       ├── neutral.csv
│       ├── positive.csv
│       ├── twitter-bird.png
│       └── word_art.html
├── docs
│   ├── ...
├── model
│   ├── create_loader.py
│   ├── model_class.py
│   ├── plot.py
│   ├── random_search.py
│   └── train_model.py
├── result
│   ├── data_visualization.xlsx
│   ├── figures
│   │   ├── ...
│   ├── predictions
│   │   ├── check_contains_emoji.ipynb
│   │   └── get_predictions.ipynb
│   └── results.csv
└── team_submission.md
``````