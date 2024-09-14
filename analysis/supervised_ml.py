import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize
import torch
from tqdm import tqdm
from pprint import pprint
import numpy as np
from itertools import combinations
from sklearn.inspection import permutation_importance
import os
import pickle
import re
import sys
import math
import collections
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from collections import defaultdict
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import BaseCrossValidator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import MeCab
from wikipedia2vec import Wikipedia2Vec
import datetime
import itertools

from stacking import train_master, MyStackingClassifer

def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


num_args = len(sys.argv)
assert num_args == 6

exp_setup = int(sys.argv[1])
print(exp_setup)

all_modes = ['cast', 'prod', 'text', 'meta']
all_combs = []
for comb_i in [1, 2, 3, 4]:
    all_combs.extend(
        list(itertools.combinations(all_modes, comb_i)))
all_combs = [list(c) for c in all_combs]
assert 0 <= exp_setup < 15
modalities = all_combs[exp_setup]
print(exp_setup, modalities)
assert modalities is not None

d1 = modification_date('extracted_actor_positions_artv.csv')
print("data", d1)
df = pd.read_csv("extracted_actor_positions_artv.csv",
                 index_col=0)\
        .drop(columns=['early_end'])

print(df.shape)
with_synopsis = df.dropna(subset=['synopsis']).reset_index(drop=True)
with_synopsis = with_synopsis.sort_values(by="datetime", ascending=True)\
    .reset_index(drop=True)
print(with_synopsis.shape)

WIKI2VEC_MODEL_USE = sys.argv[3]
if WIKI2VEC_MODEL_USE == "hist":
    wiki2vec = Wikipedia2Vec.load("../wikipedia2vec/jawiki-20061016-out-model-300")
    with open("misses2vec_hist.pkl", "rb") as f:
        misses2vec = pickle.load(f)
    print("misses2vec", " hist ", modification_date("misses2vec_hist.pkl"))
elif WIKI2VEC_MODEL_USE == "late":
    wiki2vec = Wikipedia2Vec.load("../wiki2vec2/jawiki-latest-pages-300")
    with open("misses2vec_late.pkl", "rb") as f:
        misses2vec = pickle.load(f)
    print("misses2vec", " late ", modification_date("misses2vec_late.pkl"))
else:
    raise ValueError("ok")
assert wiki2vec is not None and misses2vec is not None

PCA_TOGGLE = eval(sys.argv[5])
assert isinstance(PCA_TOGGLE, bool)
print(f"PCA TOGGLE: {PCA_TOGGLE}")


def extract_type(column_name):
    match = re.search(r'(cast|prod|meta)$', column_name)
    if match:
        return match.group()
    else:
        return 'unknown'

def get_entity_vector(wiki2vec, misses2vec, participating):
    try:
        return [wiki2vec.get_entity_vector(participating).tolist()]
    except KeyError:
        if participating in misses2vec:
            return misses2vec[participating]
        else:
            raise ValueError(f"{participating}")

columns_by_type = {}
for column in with_synopsis.columns:
    column_type = extract_type(column)
    if column_type in columns_by_type:
        columns_by_type[column_type].append(column)
    else:
        columns_by_type[column_type] = [column]
print(columns_by_type.keys())
print(modalities)

assert len(columns_by_type['prod']) == 1457
assert len(columns_by_type['cast']) == 2335
assert len(columns_by_type['meta']) == 52
assert len(columns_by_type['unknown']) == 20
del columns_by_type['meta']
del columns_by_type['unknown']

norm_embeddings_array = None
if not os.path.exists(f"pkl/text_embeddings_artv_{WIKI2VEC_MODEL_USE}.pkl"):
    print("text embeddings not found, creating..")
    from transformers import pipeline
    from sudachipy import Dictionary, SplitMode

    def extract_entities(ner_results):
        entities = []
        current_entity = {'entity': '', 'type': '', 'start': 0, 'end': 0, 'score': 0.0}

        for result in ner_results:
            if current_entity['type'] == '' or \
                result['entity'] != current_entity['type'] or \
                result['start'] != current_entity['end']:
                if current_entity['type'] != '':
                    entities.append(current_entity)
                current_entity = {
                    'entity': result['word'],
                    'type': result['entity'],
                    'start': result['start'],
                    'end': result['end'],
                    'score': result['score']
                }
            else:
                current_entity['entity'] += result['word']
                current_entity['end'] = result['end']
                current_entity['score'] = min(current_entity['score'], result['score'])

        if current_entity['type'] != '':
            entities.append(current_entity)

        return entities


    model_name = "tsmatz/xlm-roberta-ner-japanese"
    print(model_name)
    classifier = pipeline("token-classification", model=model_name)

    mecab = MeCab.Tagger('-Owakati')
    def mecab_tokenizer(text):
        parsed = mecab.parse(text)
        return parsed.split()
    vectorizer = TfidfVectorizer(tokenizer=mecab_tokenizer)
    X_corpus = vectorizer.fit_transform(with_synopsis['synopsis'].tolist())

    norm_embeddings = []
    txt_onehot_embeds = []
    entity2fails = defaultdict(lambda: -1)
    shows_failed = 0
    for tup_ind, tup in enumerate(
        tqdm(with_synopsis.itertuples(),total=len(with_synopsis))):
        my_entities = [e['entity'] for e in extract_entities(classifier(tup.synopsis))]

        tokenizer = Dictionary().create()
        morphemes = tokenizer.tokenize(tup.synopsis,SplitMode.C)
        to_use = [m.surface() for m in morphemes if
                  (m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] == "固有名詞") or
                  (m.part_of_speech()[0] == "名詞" and m.part_of_speech()[2] == "一般")]
        to_use.extend(my_entities)
        to_use = list(set(to_use))

        wiki_embeds = []
        for entity in to_use:
            ent_name = entity
            try:
                for res in get_entity_vector(wiki2vec, misses2vec, ent_name):
                    wiki_embeds.append(np.array(res))
                if entity2fails[ent_name] == -1:
                    entity2fails[ent_name] = 0
            except:
                if entity2fails[ent_name] == -1:
                    entity2fails[ent_name] = 1
                else:
                    entity2fails[ent_name] += 1
                continue
        assert len(wiki_embeds) > 0

        mean_wiki_emb = np.mean(np.stack(wiki_embeds), axis=0)

        norm_embeddings.append(mean_wiki_emb)
        txt_onehot_embeds.append(np.array(X_corpus[tup_ind].todense()).squeeze(0))

    norm_embeddings_array = np.array(norm_embeddings)
    print(norm_embeddings_array.shape)
    failed = np.sum([1 for p in entity2fails if entity2fails[p] > 0])
    ent_ok = len([k for k,v in entity2fails.items() if v == 0])
    ent_weird = len([k for k,v in entity2fails.items() if v == -1])
    print(f"text {failed} {ent_ok} {ent_weird} " +\
          f" {len(entity2fails)} {failed/len(entity2fails)}")
    print(f"{shows_failed} {len(with_synopsis)} {shows_failed/len(with_synopsis)}")
    txt_onehot_array = np.array(txt_onehot_embeds)
    print(txt_onehot_array.shape)

    with open(f"pkl/text_embeddings_artv_{WIKI2VEC_MODEL_USE}.pkl", "wb") as f:
        pickle.dump(norm_embeddings_array, f)
    with open(f"pkl/text_onehot_artv_{WIKI2VEC_MODEL_USE}.pkl", "wb") as f:
        pickle.dump(txt_onehot_array, f)
    print(f"pkl/text_embeddings_artv_{WIKI2VEC_MODEL_USE}.pkl")
    print(f"pkl/text_onehot_artv_{WIKI2VEC_MODEL_USE}.pkl")
    exit()
else:
    print("loading existing text embeddings")
    print("text", modification_date(f'pkl/text_embeddings_artv_{WIKI2VEC_MODEL_USE}.pkl'))
    assert d1 < modification_date(f'pkl/text_embeddings_artv_{WIKI2VEC_MODEL_USE}.pkl')
    with open(f"pkl/text_embeddings_artv_{WIKI2VEC_MODEL_USE}.pkl", "rb") as f:
        norm_embeddings_array = pickle.load(f)
    if norm_embeddings_array.shape[0] != df.shape[0]:
        raise ValueError("ok")
    with open(f"pkl/text_onehot_artv_{WIKI2VEC_MODEL_USE}.pkl", "rb") as f:
        txt_onehot_array = pickle.load(f)

assert norm_embeddings_array is not None and txt_onehot_array is not None
print(norm_embeddings_array.shape, txt_onehot_array.shape)

if not os.path.exists(f"pkl/cast_embeddings_cast_artv_{WIKI2VEC_MODEL_USE}.pkl") \
    or not os.path.exists(f"pkl/cast_embeddings_prod_artv_{WIKI2VEC_MODEL_USE}.pkl"):
    print("cast embeddings not found, creating..")
    for cast_type, columns_by_cast_type in columns_by_type.items():
        if cast_type not in ['cast', 'prod']:
            continue
        print(cast_type)
        cast_embeddings = []
        binary_embed_arr = []

        act2fails = {}
        for act in with_synopsis[columns_by_cast_type].columns:
            act2fails[act[:-4]] = -1

        for idx, row in tqdm(with_synopsis[columns_by_cast_type].iterrows(), total=len(with_synopsis)):
            one_list = row.where(row == 1).dropna().index.tolist()

            one_list = [o[:-4] for o in one_list if o != "drama_id"]
            partp_embeds = []
            for participating in one_list:
                try:
                    for res in get_entity_vector(wiki2vec, misses2vec, participating):
                        partp_embeds.append(np.array(res))
                    if act2fails[participating] == -1:
                        act2fails[participating] = 0
                except:
                    if act2fails[participating] == -1:
                        act2fails[participating] = 1
                    else:
                        act2fails[participating] += 1
                    continue

            try:
                assert len(partp_embeds) > 0
            except:
                import IPython
                IPython.embed()
                exit()

            mean_embedding = np.mean(np.stack(partp_embeds), axis=0)
            norm_embedding = mean_embedding
            cast_embeddings.append(norm_embedding)
            binary_embed_arr.append(row.to_numpy())

        cast_embed_arr = np.array(cast_embeddings)
        binary_embed_arr = np.array(binary_embed_arr)

        failed = np.sum([1 for p in act2fails if act2fails[p] > 0])
        act_ok = len([k for k,v in act2fails.items() if v == 0])
        act_weird = len([k for k,v in act2fails.items() if v == -1])
        print(f"{cast_type} {failed} {act_ok} {act_weird} " +\
               f" {len(act2fails)} {failed/len(act2fails)}")
        assert failed + act_ok == len(act2fails)

        with open(
            f"pkl/cast_embeddings_{cast_type}_artv_{WIKI2VEC_MODEL_USE}.pkl", "wb") as f:
            pickle.dump(cast_embed_arr, f)
        with open(f"pkl/onehot_{cast_type}_artv_{WIKI2VEC_MODEL_USE}.pkl", "wb") as f:
            pickle.dump(binary_embed_arr, f)
        print(f"pkl/cast_embeddings_{cast_type}_artv_{WIKI2VEC_MODEL_USE}.pkl")
        print(f"pkl/onehot_{cast_type}_artv_{WIKI2VEC_MODEL_USE}.pkl")
    exit()
else:
    print("loading existing cast embeddings")
    cast_embeddings_all = {}
    binary_embed_all = {}
    print("prod",
          modification_date(f'pkl/cast_embeddings_prod_artv_{WIKI2VEC_MODEL_USE}.pkl'))
    assert d1 < \
        modification_date(f'pkl/cast_embeddings_prod_artv_{WIKI2VEC_MODEL_USE}.pkl')
    with open(f"pkl/cast_embeddings_prod_artv_{WIKI2VEC_MODEL_USE}.pkl", "rb") as f:
        cast_embeddings_all['prod'] = pickle.load(f)
    with open(f"pkl/onehot_prod_artv_{WIKI2VEC_MODEL_USE}.pkl", "rb") as f:
        binary_embed_all['prod'] = pickle.load(f)
    print("cast",
          modification_date(f'pkl/cast_embeddings_cast_artv_{WIKI2VEC_MODEL_USE}.pkl'))
    with open(f"pkl/cast_embeddings_cast_artv_{WIKI2VEC_MODEL_USE}.pkl", "rb") as f:
        cast_embeddings_all['cast'] = pickle.load(f)
    with open(f"pkl/onehot_cast_artv_{WIKI2VEC_MODEL_USE}.pkl", "rb") as f:
        binary_embed_all['cast'] = pickle.load(f)


model_y_pred = np.zeros(len(with_synopsis))
model_y_proba = np.zeros(len(with_synopsis))
model_y_true = np.zeros(len(with_synopsis))
print(len(model_y_pred), len(model_y_proba), len(model_y_true))
nearest_sim_df = []

kf_res = []
y = with_synopsis['traction'].to_numpy()

EMBEDDING_MODE = sys.argv[2]
assert EMBEDDING_MODE in ["entity", "binary"]

def time_series_split(X, window_size, forecast_horizon=1):
    n_samples = len(X)
    splits = []

    for end_of_train in range(window_size, n_samples - forecast_horizon + 1):
        start_of_train = end_of_train - window_size
        start_of_test = end_of_train
        end_of_test = end_of_train + forecast_horizon

        train_index = np.arange(start_of_train, end_of_train)
        test_index = np.arange(start_of_test, end_of_test)

        splits.append((train_index, test_index))

    return splits


perm_feat_important_lis = []

dset_size = 450
custom_kfold = time_series_split(with_synopsis, dset_size, forecast_horizon=1)
for kf_ind, (train_index, test_index) in tqdm(
    enumerate(custom_kfold), total=len(custom_kfold)):
    assert len(np.setdiff1d(test_index, train_index)) == len(test_index)

    if EMBEDDING_MODE == "entity":
        PCA_COMP_CAST_FINAL = cast_embeddings_all['cast'].shape[1]
        PCA_COMP_PROD_FINAL = cast_embeddings_all['prod'].shape[1]
        PCA_COMP_TEXT_FINAL = norm_embeddings_array.shape[1]
    else:
        assert EMBEDDING_MODE == "binary"
        PCA_COMP_CAST_FINAL = binary_embed_all['cast'].shape[1]
        PCA_COMP_PROD_FINAL = binary_embed_all['prod'].shape[1]
        PCA_COMP_TEXT_FINAL = txt_onehot_array.shape[1]



    train_embs = []
    test_embs = []
    if 'text' in modalities:


        if EMBEDDING_MODE == "binary":
            txt_use_train_emb = txt_onehot_array[train_index,:]
            txt_use_test_emb = txt_onehot_array[test_index,:]
        elif EMBEDDING_MODE == "entity":
            txt_use_train_emb = norm_embeddings_array[train_index, :]
            txt_use_test_emb = norm_embeddings_array[test_index, :]
        else:
            raise ValueError("ok")

        if PCA_TOGGLE:
            pca_comp_use_text = 200
            pca = PCA(n_components=pca_comp_use_text, svd_solver='full')
            pca.fit(txt_use_train_emb)
            while np.sum(pca.explained_variance_ratio_) >= 0.6:
                pca_comp_use_text -= 5
                pca = PCA(n_components=pca_comp_use_text, svd_solver='full')
                pca.fit(txt_use_train_emb)
            if pca_comp_use_text == 0:
                pca_comp_use_text = 5

            PCA_COMP_TEXT_FINAL = pca_comp_use_text
            print("pca text DONE", pca_comp_use_text, PCA_COMP_TEXT_FINAL)
            pca_text = PCA(n_components=PCA_COMP_TEXT_FINAL, svd_solver='full')
            text_emb_train = \
                pca_text.fit_transform(txt_use_train_emb)
            text_emb_test = \
                pca_text.transform(txt_use_test_emb)
            print("pca text", txt_use_train_emb.shape,
                text_emb_train.shape,
                np.sum(pca_text.explained_variance_ratio_))
        else:
            text_emb_train = txt_use_train_emb
            text_emb_test = txt_use_test_emb

        train_embs.append(text_emb_train)
        test_embs.append(text_emb_test)


    for cast_type, columns_by_cast_type in columns_by_type.items():
        if cast_type in modalities:
            cast_embeddings = cast_embeddings_all[cast_type]


            bin_embeds = binary_embed_all[cast_type]

            if EMBEDDING_MODE == "both":
                concat_train_emb = np.concatenate(
                    (bin_embeds[train_index,:], cast_embeddings[train_index, :]), axis=1)
                concat_test_emb = np.concatenate(
                    (bin_embeds[test_index,:], cast_embeddings[test_index, :]), axis=1)
            elif EMBEDDING_MODE == "binary":
                concat_train_emb = bin_embeds[train_index,:]
                concat_test_emb = bin_embeds[test_index,:]
            elif EMBEDDING_MODE == "entity":
                concat_train_emb = cast_embeddings[train_index, :]
                concat_test_emb = cast_embeddings[test_index, :]
            else:
                raise ValueError("ok")

            if PCA_TOGGLE:
                print("before pca", concat_train_emb.shape)
                pca_comp_use = 200
                pca = PCA(n_components=pca_comp_use, svd_solver='full')
                pca.fit(concat_train_emb)
                while np.sum(pca.explained_variance_ratio_) >= 0.6:
                    pca_comp_use -= 5
                    pca = PCA(n_components=pca_comp_use, svd_solver='full')
                    pca.fit(concat_train_emb)
                if pca_comp_use == 0:
                    pca_comp_use = 5

                print("pca DONE", pca_comp_use)
                if cast_type == "cast":
                    PCA_COMP_CAST_FINAL = pca_comp_use
                elif cast_type == "prod":
                    PCA_COMP_PROD_FINAL = pca_comp_use
                else:
                    raise ValueError("ok")
                pca = PCA(n_components=pca_comp_use, svd_solver='full')
                concat_train_emb = pca.fit_transform(concat_train_emb)
                concat_test_emb = pca.transform(concat_test_emb)
                print("pca", np.sum(pca.explained_variance_ratio_))

            if exp_setup == 0 and kf_ind == 0:
                with open(f"concat_emb_{cast_type}.pkl","wb") as f:
                    pickle.dump(
                        [np.concatenate((concat_train_emb, concat_test_emb)),
                         np.concatenate((y[train_index],y[test_index]))], f)

            print(cast_type, concat_train_emb.shape)
            train_embs.append(concat_train_emb)
            test_embs.append(concat_test_emb)


    if "meta" in modalities:
        scaler = StandardScaler()
        meta_train_df = with_synopsis.iloc[train_index,
                           ["meta" in c for c in with_synopsis.columns]]

        meta_test_df = with_synopsis.iloc[test_index,
                           ["meta" in c for c in with_synopsis.columns]]

        meta_emb_train = meta_train_df.to_numpy()
        meta_emb_test = meta_test_df.to_numpy()

        if PCA_TOGGLE and exp_setup == 3:
            print("before pca", meta_emb_train.shape)
            pca_comp_use = 50
            pca = PCA(n_components=pca_comp_use, svd_solver='full')
            pca.fit(meta_emb_train)
            while np.sum(pca.explained_variance_ratio_) >= 0.6:
                pca_comp_use -= 2
                pca = PCA(n_components=pca_comp_use, svd_solver='full')
                pca.fit(meta_emb_train)
            if pca_comp_use == 0:
                pca_comp_use = 5

            print("pca DONE", pca_comp_use)

            pca = PCA(n_components=pca_comp_use, svd_solver='full')
            meta_emb_train = pca.fit_transform(meta_emb_train)
            meta_emb_test = pca.transform(meta_emb_test)
            print("pca", np.sum(pca.explained_variance_ratio_))

        train_embs.append(meta_emb_train)
        test_embs.append(meta_emb_test)

    X_train = np.concatenate(train_embs, axis=1)
    X_test = np.concatenate(test_embs, axis=1)


    print(X_train.shape)
    model = SVC(kernel="rbf", C=1, probability=True, random_state=2024)

    probas_ = model.fit(X_train, y[train_index]).predict_proba(X_test)


    probas_true = [p[1] for p in probas_]

    y_pred = model.predict(X_test)
    model_y_proba[test_index] = probas_true
    model_y_pred[test_index] = y_pred
    model_y_true[test_index] = y[test_index]

prf1_res = precision_recall_fscore_support(model_y_true[dset_size:],
                                        model_y_pred[dset_size:], pos_label=1)
print(prf1_res)


ml_outs = {
    "exp_setup": exp_setup,
    "name": "_".join(modalities),
    "embedding_mode": EMBEDDING_MODE,
    "wiki": WIKI2VEC_MODEL_USE,
    "pca": PCA_TOGGLE,
    "model_y_pred": model_y_pred,
    "model_y_proba": model_y_proba,
    "y": model_y_true,
    "mean_fpr": np.linspace(0, 1, 100),
    "pca_comp": None,
    "season_slot": with_synopsis['stationslot'].to_numpy(),
    "success_bin": with_synopsis['success_bin'].to_numpy()
}

print(f"ml_out2/ml_outs_{EMBEDDING_MODE}_{WIKI2VEC_MODEL_USE}_{exp_setup}_{PCA_TOGGLE}_artv.pkl")
with open(
    f"ml_out2/ml_outs_{EMBEDDING_MODE}_{WIKI2VEC_MODEL_USE}_{exp_setup}_{PCA_TOGGLE}_artv.pkl", "wb") as f:
    pickle.dump(ml_outs, f)

if int(sys.argv[4]) == 0:
    exit()

def get_two_year_period_with_season(row):
    year = row['year']
    season = row['month']
    station = row['station']
    if season == 10:
        season = f"{10}"
    else:
        season = f"0{season}"

    if 2003 <= year <= 2007:
        period = "2003-2007"
    elif 2008 <= year <= 2012:
        period = "2008-2012"
    elif 2013 <= year <= 2017:
        period = "2013-2017"
    elif 2018 <= year <= 2021:
        period = "2018-2021"
    else:
        raise ValueError("ok")

    return f"{station}{period}"

def get_station_with_dayweek(row):
    station = row['station']
    dayweek = row['day']
    season = row['month']
    if season == 10:
        season = f"{10}"
    else:
        season = f"0{season}"

    return f"{station}{season}"

def day_bins(name):
    if name in ['Mon', 'Tue', 'Wed']:
        return "Mon-Wed"
    elif name in ['Thu', 'Fri']:
        return 'Thu-Fri'
    elif name in ['Sat', 'Sun']:
        return 'Sat-Sun'
    else:
        raise ValueError("ok")

with_synopsis['two_year_period'] = \
    with_synopsis.apply(get_station_with_dayweek, axis=1)
print(with_synopsis['two_year_period'].unique())

from sklearn.metrics import precision_recall_fscore_support
POS_LABEL = 1
season_labels = with_synopsis['two_year_period'].to_numpy()

model2metrics_season = {}


n_bootstraps = 1000

print(EMBEDDING_MODE)

season2fail = {}
for season in tqdm(np.unique(season_labels)):
    num_rej = 0
    for _ in range(n_bootstraps):
        if season not in model2metrics_season:
            model2metrics_season[season] = []

        season_idx = np.where(season_labels == season)[0]
        season_idx = season_idx[season_idx > dset_size]
        sampled_idx = np.random.choice(season_idx,
                                       size=season_idx.size, replace=True)
        if len(season_idx) == 0:
            num_rej += 1
            continue
        spre, srec, sf1, _ = precision_recall_fscore_support(
            y[sampled_idx],
            model_y_pred[sampled_idx], average='binary', pos_label=POS_LABEL)
        acc = accuracy_score(y[sampled_idx], model_y_pred[sampled_idx])
        model2metrics_season[season].append((acc))
    print(season, num_rej)

    if num_rej == n_bootstraps:
        print("BAD")
        season2fail[season] = True
        with open(f"BAD{EMBEDDING_MODE}_{PCA_TOGGLE}_{WIKI2VEC_MODEL_USE}_{exp_setup}", "w") as f:
            f.write(f"BAD {season} {num_rej} {n_bootstraps}\n")

season2pre = {}
season2f1 = {}
season2auc = {}
for season in np.unique(season_labels):
    pre_list = []
    f1_list = []
    auc_list = []
    for res in model2metrics_season[season]:
        acc = res
        pre_list.append(np.nan)
        f1_list.append(acc)
        auc_list.append(np.nan)
    season2pre[season] = pre_list
    season2f1[season] = f1_list
    season2auc[season] = auc_list

my_dic = {'season': [], 'score': [], 'mean': [],
          'median': [], 'modality': [],
          'lower': [], 'upper': [], 'sd': []}
season2allmetrics = {
    'pre':season2pre,
    'f1':season2f1,
    'auc':season2auc}

for season in season2pre:
    for score in ['f1']:
        my_dic['season'].append(season)
        my_dic['modality'].append("".join(modalities))
        my_dic['score'].append(score)
        if season in season2fail:
            my_dic['mean'].append(np.nan)
            my_dic['median'].append(np.nan)
            my_dic['sd'].append(np.nan)
            my_dic['lower'].append(np.nan)
            my_dic['upper'].append(np.nan)
        else:
            my_dic['mean'].append(
                np.mean(season2allmetrics[score][season]))
            my_dic['median'].append(
                np.median(season2allmetrics[score][season]))
            my_dic['sd'].append(
                np.std(season2allmetrics[score][season],ddof=1))
            #75%
            my_dic['lower'].append(
                np.quantile(season2allmetrics[score][season], 0.15))
            my_dic['upper'].append(
                np.quantile(season2allmetrics[score][season], 0.85))

res_df = pd.DataFrame(my_dic)
print(res_df.head())
if EMBEDDING_MODE == "binary":
    res_df['wiki'] = "binary"
else:
    res_df['wiki'] = WIKI2VEC_MODEL_USE
res_df['pca'] = PCA_TOGGLE
if PCA_TOGGLE:
    out_name = f"season_preds_{EMBEDDING_MODE}_pca_{WIKI2VEC_MODEL_USE}_{exp_setup}.csv"
else:
    out_name = f"season_preds_{EMBEDDING_MODE}_base_{WIKI2VEC_MODEL_USE}_{exp_setup}.csv"
res_df.to_csv(out_name)

print("done")







