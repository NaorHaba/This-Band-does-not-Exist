from functools import partial
import pandas as pd


def split_to_ngram(text, n):
    word_list = text.split()
    ngram_list = []
    for i in range(len(word_list) - n):
        ngram_list.append(word_list[i:i+n])
    return ngram_list


def ngram_dist(song_lyrics, n, max_len, delta=0.5):
    # delta is the weight for length penalty
    # this function needs to be maximized
    # TODO: consider the length (maybe we would like longer and not shorter, but this way yields the best results
    ngrams = split_to_ngram(song_lyrics, n)
    consecutive_ngrams = 0
    for i in range(len(ngrams) - 1):
        if ngrams[i + 1] == ngrams[i]:
            consecutive_ngrams += 1
    ngram_repetitiveness = 1 - len(set(['_'.join(ngram) for ngram in ngrams])) / len(ngrams)
    score = 1 - ((1 - delta) * ngram_repetitiveness + delta * (len(song_lyrics.split()) / max_len))
    return score / (consecutive_ngrams + 1)


generated_data = pd.read_csv('../generated_data.csv')

max_len = max(generated_data['lyrics'].apply(lambda x: len(x.split())))
for n in range(1, 4):
    generated_data[f'{n}gram_dist'] = generated_data['lyrics'].apply(partial(ngram_dist, max_len=max_len, n=n, delta=0.8))
generated_data['score'] = generated_data.mean(axis=1)

generated_data = generated_data.sort_values(by=['score'])
print(f'Total score: {generated_data["score"].mean()}')

for i in [1, 100, 9000]:
    print(generated_data.iloc[i]['score'])
    print(generated_data.iloc[i]['lyrics'])
    print("********************")

