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
--summary_comment=gpt2_reverse_train_split_0_id_01
--train_data_file=data/all_data.csv
--output_dir=models/gpt2_forward_model
--eval_data_file=data/all_data.csv
--csv
--evaluate_during_training
--eval_creativity_blacklist
data/artists_blacklist.pickle
--eval_creativity_genres
data/genres.pickle
--num_eval_creativity
50
--eval_creativity_max_iterations
5
--eval_subsampling
0.05
--splits
0.5
0.4
0.1
--eval_split_idx
2
--train_split_idx
0
--model_name_or_path=gpt2
--model_type=gpt2
--block_size
1024
--do_train
--do_eval
--train_batch_size
1
--eval_batch_size
1
--gradient_accumulation_steps
8
--learning_rate
0.00005
--num_train_epochs
3
--logging_steps
1000
--save_steps=1000
--save_total_limit=5`

Please notice that the above script is accessing a data file that is populated with all the band records we had.
However, we process it using the parsing mechanism noted by `--csv` and according to the `--splits` and `--train_split_idx`, `--eval_splits_idx` arguments.

You can use the generate_data.py file to generate new bands. Change the batch-size parameter to decide how many bands to generate, and the save path parameter to choose a path to save the data. <br>
You can also use the evaluation.py file to perform some evaluation on generated entities (or use it as a basis for the evaluation method you want for your project). 