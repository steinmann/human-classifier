#!/usr/bin/env python3
import argparse
from statistics import mean

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import KFold, RandomizedSearchCV
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
    parser.add_argument('--tune', action='store_const', const=True, default=False,
                        help='freshly tune parameters using random search')
    args = parser.parse_args()

    # read data
    hero_info = pd.read_csv(args.info, index_col=0)
    powers = pd.read_csv(args.power)

    # merge using inner join
    hero_data = hero_info.merge(powers, left_on='name', right_on='hero_names')

    # remove heroes with undefined race
    hero_race = hero_data[hero_data['Race'] != '-']

    # engineer features
    powers = hero_race.iloc[:, 11:]

    weight = hero_race['Weight'].to_numpy()
    weight[weight == -99] = np.nan

    height = hero_race['Height'].to_numpy()
    height[height == -99] = np.nan

    gender = pd.get_dummies(hero_race['Gender'])
    eye = pd.get_dummies(hero_race['Eye color'])
    hair = pd.get_dummies(hero_race['Hair color'])
    publisher = pd.get_dummies(hero_race['Publisher'])
    skin = pd.get_dummies(hero_race['Skin color'])
    alignment = pd.get_dummies(hero_race['Alignment'])

    # defina data, label and model
    data = np.column_stack((powers, weight, height, gender, eye, hair, publisher, skin, alignment))
    label = (hero_race['Race'] == 'Human').to_numpy()
    model = xgb.XGBClassifier(max_depth=2,  # 3
                              min_child_weight=1,  # 1
                              gamma=0,  # 0
                              subsample=1,  # 1
                              colsample_bytree=1,  # 1
                              reg_alpha=0.01,  # 0
                              learning_rate=0.1,  # 0.1
                              n_estimators=100,  # 100
                              random_state=42)

    # tune hyperparameters
    if args.tune:
        params = {
            'max_depth': range(2, 8),
            'min_child_weight': range(1, 6),
            'gamma': [i / 10.0 for i in range(0, 5)],
            'subsample': [i / 10.0 for i in range(6, 11)],
            'colsample_bytree': [i / 10.0 for i in range(6, 11)],
            'reg_alpha': [i / 100.0 for i in range(0, 6)],
            'n_estimators': range(50, 150),
            'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
        }

        search = RandomizedSearchCV(model,
                                    param_distributions=params,
                                    random_state=42,
                                    n_iter=500,
                                    cv=5,
                                    n_jobs=8,
                                    return_train_score=True)

        search.fit(data, label)
        for param in search.best_params_:
            print(f'{param}: {search.best_params_[param]}')
        model = xgb.XGBClassifier(**search.best_params_)

    # get cross validated model accuracy
    accuracy = cv_accuracy(data, label, model)
    print(f'Prediction accuracy: {accuracy * 100:.2f}%')

    # train on full dataset
    model.fit(data, label)

    # save model
    model.save_model(args.model)
    print(f'Saved model to {args.model}')


if __name__ == '__main__':
    main()
