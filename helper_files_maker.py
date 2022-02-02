import argparse
import os
import pandas as pd
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('task', type=str, help='industries or blacklist')
parser.add_argument('data_path', type=str, help='Path for all data')
args = parser.parse_args()


# read and concatenate company data files
if args.task == 'concatenate_data':
    df0 = pd.read_csv('../../../../datashare/2021/data_chunk_0.csv', index_col=0,
                      usecols=['id', 'company_name', 'industry', 'text'])
    df1 = pd.read_csv('../../../../datashare/2021/data_chunk_1.csv', index_col=0,
                      usecols=['id', 'company_name', 'industry', 'text'])
    df2 = pd.read_csv('../../../../datashare/2021/data_chunk_2.csv', index_col=0,
                      usecols=['id', 'company_name', 'industry', 'text'])
    df = pd.concat([df0, df1, df2])
    # drop nan and shuffle:
    df = df.dropna().sample(frac=1).reset_index(drop=True)
    df.to_csv('data/all_data.csv', lineterminator='\n')


# make industries
if args.task == 'industries':
    for n in range(1000, 1700000, 5000):
        df = pd.read_csv(args.data_path, nrows=n, usecols=['industry'])
        industries = set(df['industry'])
        if len(industries) == 147:
            with open('data/industries.pickle', 'wb') as handle:
                pickle.dump(industries, handle)
            break


# make blacklist
elif args.task == 'blacklist':
    df = pd.read_csv(args.data_path, usecols=['company_name'])
    companies = set(df['company_name'])
    try:
        with open('data/blacklist.pickle', 'rb') as handle:
            prev_companies = handle
            companies.add(prev_companies)
        with open('data/blacklist.pickle', 'wb') as handle:
            pickle.dump(companies, handle)
    except:
        with open('data/blacklist.pickle', 'wb') as handle:
            pickle.dump(companies, handle)
