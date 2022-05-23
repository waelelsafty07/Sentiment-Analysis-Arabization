from copyreg import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from preprocessing import preprocessing
from modeling import modeling, random_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from test_classifier import train_n_test_classifier

# Arabic Sentiment Analysis Dataset - SS2030.csv has 4252 rows in reality
# but we are only loading / previewing the first 1000 rows

def build_and_save_model(train_data, test_data, val_data): # run one first time for building model

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_df=0.5, stop_words=None, use_idf=True)

    train_data_features = vectorizer.fit_transform(train_data['normalized_text'].values.astype('U'))
    val_data_features = vectorizer.transform(val_data['normalized_text'].values.astype('U'))
    test_data_features = vectorizer.transform(test_data['normalized_text'].values.astype('U'))

    print(train_data_features.shape, val_data_features.shape, test_data_features.shape)

    rand_seed = 0
    output = 'Sentiment' # output label column
    rf = RandomForestClassifier(n_estimators=100, random_state=rand_seed)

    train_n_test_classifier(rf, train_data_features, train_data[output],
                            val_data_features, val_data[output])

    pickle.dump(rf, open('experiment_folder/rf.pkl', 'wb'))



def predict_multi_level(X, neu_vectorizer, neu_clf, vectorizer, clf):
    #return clf.predict(vectorizer.transform(X))
    neu_y_pred = neu_clf.predict(neu_vectorizer.transform(X))
    if len(X[neu_y_pred == 'NonNeutral']) > 0:
        y_pred = clf.predict(vectorizer.transform(X[neu_y_pred == 'NonNeutral'])) # classify non neutral into positive or negative
        neu_y_pred[neu_y_pred == 'NonNeutral'] = y_pred
    
    final_y_pred = neu_y_pred
    return final_y_pred

def use_model(test_data, neu_vectorizer, neu_mlp, vectorizer, mnb):
    rf = pickle.load(open('experiment_folder/rf.pkl', 'rb'))

    X = test_data.dropna()['normalized_text'].values
    y = test_data.dropna()['sentiment'].values
    pred_y = predict_multi_level(X, neu_vectorizer, neu_mlp, vectorizer, mnb)
    
    print('accuracy_score: ')
    print(accuracy_score(y, pred_y))

    print('f1_score: ')
    print(f1_score(y, pred_y, average='macro'))

    pass 



def core_process():
    data = pd.read_csv('input/Arabic Sentiment Analysis Dataset - SS2030.csv', delimiter=',')

    data = preprocessing(data)

    train_data, val_data, test_data = modeling(data)

    build_and_save_model(train_data, test_data, val_data)

    use_model(test_data, )