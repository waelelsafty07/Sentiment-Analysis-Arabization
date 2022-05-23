import pandas as pd
from arnormalize import normalize_arabic_text

def prepare_text(data): 
    data['normalized_text'] = normalize_arabic_text(data['text'])
    return data

def preprocessing(data):
    data = data.apply(prepare_text, axis=1)

    data = data.drop_duplicates('normalized_text').reset_index(drop=True)  # remove duplicate data

    min_class = min(len(data[data['Sentiment'] == 1]),len(data[data['Sentiment'] == 0]))

    sampled_positive = data[data['Sentiment'] == 1].sample(n=min_class)
    sampled_negative = data[data['Sentiment'] == 0].sample(n=min_class)

    data_dist = [len(sampled_positive), len(sampled_negative)]

    balanced_data = sampled_positive.append(sampled_negative).sample(frac=1).reset_index(drop=True)

    return balanced_data