#!/usr/bin/env python
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k, reciprocal_rank
import argparse as ap
import itertools
import numpy as np
import operator
import pandas as pd


def model(df, params, u=None, i=None):
    state = np.random.RandomState(params['seed'])
    data = Dataset()
    data.fit(
        df['userID'].unique(),
        df['poiID'].unique(),
        user_features=u[1] if u is not None else None,
        item_features=i[1] if i is not None else None
    )

    if u is not None:
        user_features_iterable = map(lambda l: (l[0], l[1]), u[0].iteritems())
        user_features = data.build_user_features(user_features_iterable, normalize=False)
    else:
        user_features = None

    if i is not None:
        item_features_iterable = map(lambda l: (l[0], [l[1]]), i[0].iteritems())
        item_features = data.build_item_features(item_features_iterable, normalize=False)
    else:
        item_features = None

    ratings, weights = data.build_interactions(df[['userID', 'poiID']].itertuples(index=False, name=None))

    train, test = random_train_test_split(
        ratings, test_percentage=params['test'], random_state=state
    )

    lfm = LightFM(
        no_components=params['f'],
        learning_rate=params['lr'],
        loss=params['loss'],
        user_alpha=params['alpha'],
        random_state=state
    )
    lfm.fit(train, epochs=params['epochs'], user_features=user_features, item_features=item_features)

    return {
        'pr-train': 100.0 * precision_at_k(
            lfm, train, k=params['k'], user_features=user_features, item_features=item_features
        ).mean(),
        'mrr-train': 100.0 * reciprocal_rank(
            lfm, train, user_features=user_features, item_features=item_features
        ).mean(),
        'pr-test': 100.0 * precision_at_k(
            lfm, test, k=params['k'], user_features=user_features, item_features=item_features
        ).mean(),
        'mrr-test': 100.0 * reciprocal_rank(
            lfm, test, user_features=user_features, item_features=item_features
        ).mean()
    }


parser = ap.ArgumentParser(description='User Personas Recommender',  formatter_class=ap.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d',  type=str, help='Flickr User-POI Visits file', default=ap.SUPPRESS)
parser.add_argument('-u', type=str, help='Profiles file', default=ap.SUPPRESS)
parser.add_argument('--test', type=float, help='Test percentage', default=0.2)
parser.add_argument('--seed', type=int, help='Random seed', default=2020)
parser.add_argument('--f', type=int, help='Number of features', default=20)
parser.add_argument('--lr', type=float, help='Learning rate', default=0.05)
parser.add_argument('--a', type=float, help='L2 regularization parameter (user features)', default=0.005)
parser.add_argument('--loss', type=str, help='Loss function', default='bpr')
parser.add_argument('--epochs', type=int, help='Number of epochs', default=20)
parser.add_argument('--k', type=int, help='Recommendation list size', default=3)

args = parser.parse_args()

params = {
    'test': args.test,
    'seed': args.seed,
    'f': args.f,
    'lr': args.lr,
    'loss': args.loss,
    'alpha': args.a,
    'epochs': args.epochs,
    'k': args.k
}

themes = ['Historical', 'Cultural', 'Museum', 'Structure', 'Park', 'Sport', 'Amusement', 'Shopping', 'Beach',
          'Entertainment', 'Transport', 'Palace', 'Education', 'Architectural', 'Zoo', 'Religious', 'Precinct',
          'Religion', 'Building']


interactions = pd.read_csv(args.d, sep=';')
pois = interactions[['poiID','poiTheme']].drop_duplicates()
pois = pois.set_index('poiID')
features = pd.read_csv(args.u, index_col=0)
cols = features.columns.tolist()

fmt_str = 'Precision: Train {pr-train:.2f}%, Test {pr-test:.2f}%\nMRR: Train {mrr-train:.2f}%, Test {mrr-test:.2f}%\n'

print('LightFM Model (Baseline)\n------------------------')
print(fmt_str.format(**model(interactions, params)))

print('LightFM Model + User Personas\n-----------------------------')
print(fmt_str.format(**model(
    interactions, params, (features.apply(lambda l: l.to_dict(), axis=1), cols), (pois['poiTheme'], themes)
)))