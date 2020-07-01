#!/usr/bin/env python
from collections import defaultdict
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import argparse as ap
import numpy as np
import pandas as pd

culture = ['Historical', 'Cultural', 'Museum', 'Park', 'Amusement', 'Entertainment', 'Palace', 'Architectural',
           'Religious', 'Religion', 'Building']


def centrality(data):
    f = data[['userID', 'poiID', 'poiTheme', 'seqID']].drop_duplicates()
    f['cultural'] = f['poiTheme'].apply(lambda l: 1 if l in culture else 0)
    f_ret = f[['userID', 'cultural']].groupby('userID').agg(sum)
    f_ret['log_centrality'] = f_ret['cultural'].apply(lambda l: np.log(l) if l > 0.0 else 0.0)
    min_val = f_ret['log_centrality'][f_ret['log_centrality'] > 0].min()
    diff = f_ret['log_centrality'].max() - min_val
    f_ret['log_centrality'] = 1.0 + 4.0*(f_ret['log_centrality'] - min_val) / diff
    min_val = f_ret['log_centrality'].min()
    diff = f_ret['log_centrality'].max() - min_val
    return 1.0 + 4.0*(f_ret['log_centrality'] - min_val) / diff


def visit_frequency(data):
    vf = data[['userID', 'poiID', 'seqID']].drop_duplicates()
    vf_ret = vf.groupby(['userID'])[['poiID', 'seqID']].nunique()
    vf_ret['freq'] = vf_ret['poiID'] / vf_ret['seqID']
    min_freq = vf_ret['freq'].min()
    diff = vf_ret['freq'].max() - min_freq
    return 1.0 + 4.0 * (vf_ret['freq'] - min_freq) / diff


def knowledge(data):
    p = data[data['poiTheme'].isin(culture)][['poiID', 'seqID']].drop_duplicates()
    p_v = pd.DataFrame(p['poiID'].value_counts())
    p_v['log_visits'] = np.log(p_v['poiID'])
    max_log_visits = p_v['log_visits'].max()
    p_v['freq'] = 1.0 + 4.0 * (1.0 - p_v['log_visits'] / max_log_visits)
    u_v = data[['userID', 'poiID']].drop_duplicates()
    u_v_f = u_v.merge(pd.DataFrame(p_v['freq']), how='outer', left_on='poiID', right_index=True)
    u_v_f['freq'] = u_v_f['freq'].fillna(0.0)

    min_freq = u_v_f['freq'].min()
    diff = u_v_f['freq'].max() - min_freq
    u_v_f['freq'] = 1.0 + 4.0*(u_v_f['freq'] - min_freq)/diff

    return u_v_f[['userID', 'freq']].groupby('userID').max()['freq']


def process_dict(dt):
    dd = defaultdict(list)
    for i in dt:
        for k, v in i.items():
            dd[k].append(v)

    return dict(dd)


def process_time(dt):
    dd = process_dict(dt)
    for k, v in dd.items():
        dd[k] = process_dict(v)

    l = []
    for v in dict(dd).values():
        for vl in v.values():
            l.append((max(vl)-min(vl)))

    ret = [i for i in l if i]

    return np.mean(ret) if len(ret) else 0.0


def visit_duration(data):
    d = data[data['poiTheme'].isin(culture)][['userID', 'poiID', 'dateTaken', 'seqID']]
    d['comb'] = d.apply(lambda l: {l[3]: {l[1]: l[2]}}, axis=1)
    d_ret = d[['userID', 'comb']].groupby('userID').agg(list)
    v_dur_all = d_ret['comb'].apply(lambda l: process_time(l))
    u = df[['userID']].groupby('userID').count()
    u['dur'] = v_dur_all
    u['dur'] = u['dur'].fillna(0.0)
    u['dur'] = u['dur'].apply(lambda l: np.log(l) if l else 0.0)
    min_log = u['dur'].min()
    diff = u['dur'].max() - min_log
    return 1.0 + 4.0 * (u['dur'] - min_log) / diff


parser = ap.ArgumentParser(description='Personas Builder')
parser.add_argument('-i', '--input', type=str, help='Flickr User-POI Visits file (input)')
parser.add_argument('-o', '--output', type=str, help='Profiles file (output)')

args = parser.parse_args()

df = pd.read_csv(args.input, sep=';')

# Centrality
centr = centrality(df)

# Frequency of visits
visit_freq = visit_frequency(df)

# Visiting Knowledge
u_know = knowledge(df)

# Duration
vis_dur = visit_duration(df)

users = pd.DataFrame()
users['centrality'] = centr
users['frequency'] = visit_freq
users['knowledge'] = u_know
users['duration'] = vis_dur

personas = pd.DataFrame({
 'type': ['purposeful', 'sightseeing', 'incidental', 'serendipitous', 'casual'],
 'centrality': [4.5, 4.0, 2.0, 1.0, 3.5],
 'frequency': [3.0, 3.0, 1.0, 3.0, 4.0],
 'knowledge': [4.0, 3.0, 1.0, 4.5, 2.0],
 'duration': [4.5, 4.0, 2.0, 4.0, 2.0]
}).set_index('type')

u_per = pd.DataFrame(index=users.index, columns=personas.index)

for i in users.iterrows():
    u_per.ix[i[0]]['purposeful'] = 1.0 / (1.0 + euclidean(i[1], personas.ix['purposeful']))
    u_per.ix[i[0]]['sightseeing'] = 1.0 / (1.0 + euclidean(i[1], personas.ix['sightseeing']))
    u_per.ix[i[0]]['incidental'] = 1.0 / (1.0 + euclidean(i[1], personas.ix['incidental']))
    u_per.ix[i[0]]['serendipitous'] = 1.0 / (1.0 + euclidean(i[1], personas.ix['serendipitous']))
    u_per.ix[i[0]]['casual'] = 1.0 / (1.0 + euclidean(i[1], personas.ix['casual']))

u_per = u_per.div(u_per.sum(axis=1), axis=0)
u_per.to_csv(args.output)
