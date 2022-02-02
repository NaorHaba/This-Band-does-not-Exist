import argparse
import os
import pandas as pd
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('task', type=str, help='genres or blacklist')
parser.add_argument('data_path', type=str, help='Path for all data')
args = parser.parse_args()


# read and concatenate train and test data files
if args.task == 'concatenate_data':
    train = pd.read_csv('data/train.csv', usecols=['Artist', 'Song', 'Genre', 'Language', 'Lyrics'])
    test = pd.read_csv('data/test.csv', usecols=['Artist', 'Song', 'Genre', 'Lyrics'])
    train = train[train.Language == 'en'].drop(columns=['Language'])
    data = pd.concat([train, test])
    # drop nan (and shuffle):
    data = data.dropna().sample(frac=1).reset_index(drop=True)
    data.to_csv('data/all_data.csv')


# make genres
if args.task == 'genres':
    data = pd.read_csv(args.data_path, usecols=['Genre'])
    genres = set(data['Genre'])
    with open('data/genres.pickle', 'wb') as handle:
        pickle.dump(genres, handle)


# make blacklist
elif args.task == 'blacklist':
    data = pd.read_csv(args.data_path, usecols=['Artist'])
    artists = set(data['Artist'])
    try:
        with open('data/artists_blacklist.pickle', 'rb') as handle:
            prev_artists = handle
            artists.add(prev_artists)
        with open('data/artists_blacklist.pickle', 'wb') as handle:
            pickle.dump(artists, handle)
    except:
        with open('data/artists_blacklist.pickle', 'wb') as handle:
            pickle.dump(artists, handle)
