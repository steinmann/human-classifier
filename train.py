#!/usr/bin/env python3
import argparse
from statistics import mean

import pandas as pd
import numpy as np
import xgboost as xgb

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence


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

    numerical_features = []
    for feature in ['Weight', 'Height']:
        num = hero_race[feature].to_numpy()
        num[num == -99] = np.nan
        numerical_features.append(num)

    categorical_features = []
    for feature in ['Gender', 'Eye color', 'Hair color', 'Publisher', 'Skin color', 'Alignment']:
        categorical_features.append(pd.get_dummies(hero_race[feature]))

    # define data, label and model
    data = np.column_stack([powers] + numerical_features + categorical_features)
    label = (hero_race['Race'] == 'Human').to_numpy()
    model = xgb.XGBClassifier(max_depth=5,  # 3
                              min_child_weight=1,  # 1
                              gamma=2.0890269956665937,  # 0
                              subsample=1,  # 1
                              colsample_bytree=1,  # 1
                              reg_alpha=0,  # 0
                              learning_rate=0.12187163648342948,  # 0.1
                              n_estimators=50,  # 100
                              random_state=42)

    # tune hyperparameters
    if args.tune:
        # define parameter space and tuning objective
        space = [Integer(1, 10, name='max_depth'),
                 Integer(1, 5, name='min_child_weight'),
                 Real(0.0, 5, name='gamma'),
                 Real(0.5, 1, name='subsample'),
                 Real(0.5, 1, name='colsample_bytree'),
                 Real(0, 0.05, name='reg_alpha'),
                 Real(0.0001, 0.3, name='learning_rate'),
                 Integer(50, 150, name='n_estimators')]

        @use_named_args(space)
        def objective(**params):
            model.set_params(**params)
            return -1 * cv_accuracy(data, label, model)

        # tune parameters
        model_gp = gp_minimize(objective, space, n_calls=500, random_state=42, verbose=True)

        # create dictionary of optimized parameters
        param_names = [param.name for param in space]
        opt_params = dict(zip(param_names, model_gp.x))

        # create model with tuned parameters
        model = xgb.XGBClassifier(**opt_params)

        # plot convergence of optimization
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_convergence(model_gp, ax=ax)
        fig.savefig('convergence.png')

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
