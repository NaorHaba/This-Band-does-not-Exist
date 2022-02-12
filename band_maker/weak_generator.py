import os.path
import pickle

from gensim import corpora
from gensim.models import TfidfModel
import pandas as pd
import random

from utils import clean_text


class WeakFakeGenerator:
    def __init__(self, data_path, save_cache=None):
        if save_cache and os.path.isfile(save_cache):
            print("Loading tfidf model from path...", end=' ')
            with open(save_cache, 'rb') as tfidf_file:
                self.model, self.corpus, self.dictionary, self.industries = pickle.load(tfidf_file)
            print("done.")
        else:
            print("Creating tfidf model...", end=' ')
            data = pd.read_csv(data_path, lineterminator='\n', usecols=['company_name', 'industry', 'text'])
            self.model, self.corpus, self.dictionary, self.industries = self.create_tfidf_model(data)
            print("done.")
            if save_cache:
                print("Saving tfidf model to path...", end=' ')
                with open(save_cache, 'wb') as tfidf_file:
                    pickle.dump((self.model, self.corpus, self.dictionary, self.industries), tfidf_file)
                print("done.")
        self.industry_to_index = {ind: i for i, ind in enumerate(set(self.industries))}

    @staticmethod
    def create_tfidf_model(data):
        print('making industry documents... ', end='')
        group = data.groupby(['industry'])['text'].apply(lambda x: ' '.join(x)).reset_index()
        texts = group.lyrics
        industries = group.genre.to_list()
        texts = [clean_text(text).split() for text in texts]
        print('done.')
        print('making dictionary... ', end='')
        dictionary = corpora.Dictionary(texts)
        dictionary.save('dictionary.dict')
        print('done.')
        print('making corpus... ', end='')
        corpus = [dictionary.doc2bow(text) for text in texts]
        corpora.MmCorpus.serialize('corpus.mm', corpus)
        print('done.')
        return TfidfModel(corpus), corpus, dictionary, industries

    def generate_from_industry(self, industry, tfidf):
        ind_idx = self.industry_to_index[industry]
        words = [w[0] for w in self.model[self.corpus[ind_idx]]]
        if tfidf:
            probs = [p[1] for p in self.model[self.corpus[ind_idx]]]
        else:
            probs = None
        k = int(200 - random.uniform(0, 180**(2/3))**(3/2))
        top_k_words = random.choices(words, weights=probs, k=k)
        text = ' '.join([self.dictionary[w] for w in top_k_words])
        return text

    def generate(self, data, tfidf=True, save_path=None):
        new_data = data.copy()
        new_data['text'] = new_data.genre.apply(lambda ind: self.generate_from_industry(ind, tfidf))
        if save_path:
            new_data.to_csv(save_path)
        return new_data
