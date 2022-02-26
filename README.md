# This-Band-does-not-Exist

This repository was created as a final project to the Technion course _Natural Language Processing (NLP)_. <BR>
In this project we followed the work of [this-word-does-not-exist](https://github.com/turtlesoupy/this-word-does-not-exist)
to create a framework which can be used to generate fake bands. <BR>
The framework includes the train, evaluation and generation scripts and is built in an adaptive way, capable of
recreating similar results even on other domains.

The use of the scripts is described as following: <BR> <BR>
To train the model, we will use the main script in _train.py_ - arguments to control the script are described
in the argparse object and can be accessed using the --help command. <BR>
On our project we trained using the following call: <BR>
`python3 train.py
--summary_comment=gpt2_train_split_0
--train_data_file=data/all_data.csv --output_dir=models/gpt2_forward_model
--eval_data_file=data/all_data.csv --csv --eval_creativity_blacklist data/company_blacklist.pickle
--eval_creativity_industries data/industries.pickle --eval_subsampling 0.2
--splits 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 --eval_split_idx 9 --train_split_idx 0
--model_name_or_path=gpt2 --model_type=gpt2 --block_size 512 --do_train --do_eval
--train_batch_size 2 --eval_batch_size 2 --gradient_accumulation_steps 4 --learning_rate 0.00005
--num_train_epochs 3 --logging_steps 10 --save_steps=2500 --save_total_limit=10`

Please notice that the above script is accessing a data file that is populated with all the band records we had.
However, we process it using the parsing mechanism noted by `--csv` and according to the `--splits` and `--train_split_idx`, `--eval_splits_idx` arguments.
This leaves us with ??? (first) records for train and ??? (last) records for evaluation from the ??? million records we had.

***CHANGE THE FOLLOWING***:

To create a dataset of generated bands for the model we trained, we used the code cells from
the _create_classification_datasets.ipynb_ that create an object from the CompanyGenerator class and
use its _generate_companies_ function.<BR>
In this notebook, using the dataset we generated, we created 6 more datasets used for the generation evaluation part: <BR>
- Random generated train & test files - train & test files containing together a sample of 50,000 real companies and 50,000 "artificial" fake
  companies generated by sampling randomly words from a corpus according to industry.
- Weighted random generated train & test files by tf-idf - similar to the above only now we sample according
  to a distribution achieved by using tf-idf scores for words in each industry corpus.
- Model generated train & test files - similar to the above only now the generated companies are coming from the model we trained on.

The content of these datasets is farther described in the attached report to the project. <BR>
The first 2 datasets described above are created using an object from the WeakFakeGenerator class.


Lastly, to run the generation evaluation part, by using the comparison of the results of a model trained with weak
generated companies when evaluated on a weak dataset vs our model generated dataset, we used the _fake_classifier_ notebook.
We used it with several train & test configurations to achieve the results we described in the report.