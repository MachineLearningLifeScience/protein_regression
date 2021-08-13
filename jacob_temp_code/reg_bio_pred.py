# %%
import pickle
from helper_functions import *
from collections import defaultdict
import numpy as np



# %%
# get var
names = ['blat', 'mth3', 'timb', 'calm', 'brca']
names = [names[0]]
for name in names:
    if name=='blat':
        name = name.upper()
    offset = 4
    cv_folds = 10
    split_by_DI = False

    # get stuff on var
    pos_per_fold = pos_per_fold_assigner(name.lower())
    df = pickle.load( open('data/'+name+'_data_df.pkl', "rb" ) )
    query_seqs = df['seqs'][0]
    assay_df = df.dropna(subset=['assay']).reset_index(drop=True)
    y = assay_df['assay']

    X = onehot_(assay_df['seqs'].values)

    # regular cv
    output_regular = regularCV_pred(X, y, cv_folds)
    print(output_regular[0], output_regular[1])
    # Bio
    train_indices, val_indices, test_indices = positional_splitter(assay_df, query_seqs, val=False, offset = offset, pos_per_fold = pos_per_fold, 
                                                split_by_DI = split_by_DI)

    output_bio = pred_func(X, y, train_indices, test_indices, seed=42)
    print(output_bio[0], output_bio[1])

    results_dict = {}
    results_dict['mse_reg'] = output_regular[0]
    results_dict['mse_bio'] = output_bio[0]
    
    pickle.dump(results_dict, open('results/onehot_unirep_results/'+name.lower()+'_onehot_results.pkl', 'wb'))
print(output_regular[0])
# %%
import pickle
from helper_functions import *
from collections import defaultdict
import numpy as np

names = ['blat', 'mth3', 'timb', 'calm', 'brca']
for name in names:
    if name=='blat':
        name = name.upper()
    offset = 4
    cv_folds = 10
    split_by_DI = False

    # get stuff on var
    pos_per_fold = pos_per_fold_assigner(name.lower())
    df = pickle.load( open('data/'+name+'_data_df.pkl', "rb" ) )
    query_seqs = df['seqs'][0]
    assay_df = df.dropna(subset=['assay'])
    y = assay_df['assay'].reset_index(drop=True)

    if name.lower()=='blat':
        filename = 'VAE_D2021610T154_BLAT_30dim_final_results_dict.pkl'
    if name.lower()=='brca':
        filename = 'VAE_D2021610T154_brca_30dim_final_results_dict.pkl'
    if name.lower()=='mth3':
        filename = 'VAE_D2021610T154_mth3_30dim_final_results_dict.pkl'
    if name.lower()=='timb':
        filename = 'VAE_D2021610T154_timb_30dim_final_results_dict.pkl'
    if name.lower()=='calm':
        filename = 'VAE_D2021610T155_calm_30dim_final_results_dict.pkl'

    results = pickle.load(open('final_results/final/'+filename, 'rb'))
    X = results['encoded_mu'][-1][assay_df.index[0]:]

    # regular cv
    output_regular = regularCV_pred(X, y, cv_folds)
    print(output_regular[0], output_regular[1])
    # Bio
    train_indices, val_indices, test_indices = positional_splitter(assay_df, query_seqs, val=False, offset = offset, pos_per_fold = pos_per_fold, 
                                                split_by_DI = split_by_DI)

    output_bio = pred_func(X, y, train_indices, test_indices, seed=42)
    print(output_bio[0], output_bio[1])

    #results_dict = {}
    #results_dict['mse_reg'] = output_regular[0]
    #results_dict['mse_bio'] = output_bio[0]

    #pickle.dump(results_dict, open('results/onehot_unirep_results/'+name.lower()+'_VAE_results.pkl', 'wb'))



# %%
import pickle
from helper_functions import *
from collections import defaultdict
import numpy as np

names = ['blat', 'mth3', 'timb', 'calm', 'brca']
names = ['calm', 'brca']
for name in names:
    if name=='blat':
        name = name.upper()
    offset = 4
    cv_folds = 10
    split_by_DI = False

    # get stuff on var
    pos_per_fold = pos_per_fold_assigner(name.lower())

    df = pickle.load( open( 'data/'+name.lower()+'_seq_reps_n_phyla.pkl', "rb" ) )
    query_seqs = df['seqs'][0]
    assay_df = df.dropna(subset=['assay']).reset_index(drop=True)

    X = np.vstack(assay_df['protbert_mean'])
    y = assay_df['assay']

    # regular cv
    output_regular = regularCV_pred(X, y, cv_folds)
    print(output_regular[0], output_regular[1])
    # Bio
    train_indices, val_indices, test_indices = positional_splitter(assay_df, query_seqs, val=False, offset = offset, pos_per_fold = pos_per_fold, 
                                                split_by_DI = split_by_DI)

    output_bio = pred_func(X, y, train_indices, test_indices, seed=42)
    print(output_bio[0], output_bio[1])

    results_dict = {}
    #results_dict['mse_reg'] = output_regular[0]
    #results_dict['mse_bio'] = output_bio[0]

    #pickle.dump(results_dict, open('results/onehot_unirep_results/'+name.lower()+'_avgtransformer_results.pkl', 'wb'))
