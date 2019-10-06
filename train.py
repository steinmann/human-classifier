#!/usr/bin/env python3
import argparse
from statistics import mean

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def cv_accuracy(data, labels, model):
    # return mean accuracy from five-fold cross validation
    kf = KFold(n_splits=5)
    kf.get_n_splits(data)
    accuracy_scores = []
    for train_index, test_index in kf.split(data):
        # split training data and labels
        train_data = data[train_index, :]
        train_labels = labels[train_index]
        # fit model
        model.fit(train_data, train_labels)
        # predict test data
        test_data = data[test_index, :]
        test_labels = labels[test_index]
        predictions = model.predict(test_data)
        # transform predictions to binary
        predictions[predictions > 0.5] = 1
        predictions[predictions <= 0.5] = 0
        # save accuracy score
        accuracy_scores.append(accuracy_score(test_labels, predictions))
    return mean(accuracy_scores)


def main():
    # set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--info', nargs='?', required=True,
                        help='csv file containing general superhero info')
    parser.add_argument('--power', nargs='?', required=True,
                        help='csv file containing info on superhero powers')
    parser.add_argument('--model', nargs='?', default='human_classifier.model',
                        help='name of the model (optional)')
    args = parser.parse_args()

    # read data
    hero_info = pd.read_csv(args.info, index_col=0)
    powers = pd.read_csv(args.power)

    # merge using inner join
    hero_data = hero_info.merge(powers, left_on='name', right_on='hero_names')

    # remove heroes with undefined race
    hero_race = hero_data[hero_data['Race'] != '-']

    # use only superpowers for learning
    powers = hero_race.iloc[:, 11:].to_numpy()

    # create labels
    is_human = (hero_race['Race'] == 'Human').to_numpy()

    # define model
    model = xgb.XGBClassifier(random_state=42)

    # get cross validated model accuracy
    accuracy = cv_accuracy(powers, is_human, model)
    print(f'Prediction accuracy: {accuracy * 100:.2f}%')

    # train on full dataset
    model.fit(powers, is_human)

    # save model
    model.save_model(args.model)
    print(f'Saved model to {args.model}')


if __name__ == '__main__':
    main()
