# %%
from pnet import pnet_loader, Pnet, PnetOG, ReactomeNetworkOG
from util import util, sankey_diag

import torch
import seaborn as sns
import pandas as pd
import numpy as np
import random
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import os
import torch.nn.functional as F
import torch.nn as nn

%load_ext autoreload
%autoreload 2

# %% [markdown]
# Generate small test dataset

# %%
# rna_ext_val = pd.read_csv('/mnt/disks/pancan/data/mel_dfci_2019/data_RNA_Seq_expression_tpm_all_sample_Zscores.txt',
#                           delimiter='\t').set_index('Hugo_Symbol').T.drop('Entrez_Gene_Id').dropna(axis=1)
# cna_ext_val = pd.read_csv('/mnt/disks/pancan/data/mel_dfci_2019/data_CNA.txt',
#                           delimiter='\t').set_index('Hugo_Symbol').T.dropna(axis=1)
# ext_val = pd.read_csv('/mnt/disks/pancan/data/mel_dfci_2019/data_clinical_sample.txt',
#                              delimiter='\t').set_index('Sample Identifier').iloc[4:]
# important_genes = list(pd.read_csv('/mnt/disks/pancan/m1000/cancer_genes.txt')['genes'].values)
# joint_genes = list(set(important_genes).intersection(list(rna_ext_val.columns), list(cna_ext_val.columns)))
# gene_list = random.sample(joint_genes, 500)
# random_genes_a = list(rna_ext_val.sample(5, axis=1).columns)
# random_genes_b = list(cna_ext_val.sample(5, axis=1).columns)
# joint_samples = list(rna_ext_val.sample(20).join(cna_ext_val, rsuffix='_cna', how='inner').index)
# random_samples_a = list(rna_ext_val.sample(5, axis=0).index)
# random_samples_b = list(cna_ext_val.sample(5, axis=0).index)
# random_samples_c = list(cna_ext_val.sample(5, axis=0).index)
# random_samples_d = list(cna_ext_val.sample(5, axis=0).index)
# test_rna = rna_ext_val.loc[joint_samples+random_samples_a][joint_genes+random_genes_a].copy().drop_duplicates()
# test_cna = cna_ext_val.loc[joint_samples+random_samples_b][joint_genes+random_genes_b].copy().drop_duplicates()
# test_add = ext_val.loc[joint_samples+random_samples_c][['Purity', 'Ploidy']].copy().drop_duplicates()
# test_y = ext_val.loc[joint_samples+random_samples_d][['Heterogeneity']].copy().drop_duplicates()
# test_rna.reset_index(inplace=True)
# test_cna.reset_index(inplace=True)
# test_add.reset_index(inplace=True)
# test_y.reset_index(inplace=True)
# test_rna.rename(columns={'index': 'sample_id'}, inplace=True)
# test_cna.rename(columns={'index': 'sample_id'}, inplace=True)
# test_add.rename(columns={'Sample Identifier': 'sample_id'}, inplace=True)
# test_y.rename(columns={'Sample Identifier': 'sample_id'}, inplace=True)
# test_rna.to_csv('../data/test_data/rna.csv', index=False)
# test_cna.to_csv('../data/test_data/cna.csv', index=False)
# test_add.to_csv('../data/test_data/add.csv', index=False)
# test_y.to_csv('../data/test_data/y.csv', index=False)
# with open('../data/test_data/gene_sublist.txt', 'wb') as fp:
#     pickle.dump(gene_list, fp)

# %% [markdown]
# Read test data

# %%
test_rna = pd.read_csv('../data/test_data/rna.csv').set_index('sample_id')
test_cna = pd.read_csv('../data/test_data/cna.csv').set_index('sample_id')
test_add = pd.read_csv('../data/test_data/add.csv').set_index('sample_id')
test_y = pd.read_csv('../data/test_data/y.csv').set_index('sample_id')

with open('../data/test_data/gene_sublist.txt', 'rb') as fp:
    gene_list = pickle.load(fp)

# %%
genetic_data = {'rna': test_rna, 'cna': test_cna}

# %%
train_dataset, test_dataset = pnet_loader.generate_train_test(genetic_data,
                                                              test_y, 
                                                              additional_data=test_add,
                                                              test_split=0.2,
                                                              gene_set=gene_list,
                                                              collinear_features=2)

# %%
assert set(gene_list) == set(train_dataset.genes), 'Training dataset expected to have the same gene set as in file'
assert train_dataset.genes == list(train_dataset.input_df.columns)[:500], 'Training data genes should be ordered \
                                                                            as stored in the genes variable'
assert train_dataset.input_df.shape == torch.Size([16, 1000]), 'Input DataFrame expected to be a of size\
                                                        [16, 1000], got: {}'.format(train_dataset.input_df.shape)
assert train_dataset.x.shape == torch.Size([16, 1000]), 'Small train dataset expected to be a tensor of size\
                                                        [16, 1000], got: {}'.format(train_dataset.x.shape)
assert train_dataset.y.shape == torch.Size([16, 1]), 'Small train dataset expected to be a tensor of size\
                                                        [16, 1], got: {}'.format(train_dataset.y.shape)


# %%
train_loader, val_loader = pnet_loader.to_dataloader(train_dataset, test_dataset, 64)

# %%
test_y_bin = test_y.apply(lambda x: round(2*x)).astype(int)

# %%
canc_genes = list(pd.read_csv('../../pnet_database/genes/cancer_genes.txt').values.reshape(-1))

# %%
class_weights = util.get_class_weights(torch.tensor(test_y_bin.values).view(-1))
model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(genetic_data,
                                                                         test_y_bin,
                                                                         verbose=True,
                                                                         early_stopping=True,
                                                                         epochs=10,
                                                                         batch_size=10,
                                                                         loss_weight=class_weights,
                                                                         loss_fn=nn.BCEWithLogitsLoss(reduce=None),
                                                                         gene_set = canc_genes
                                                                        )

x_train = train_dataset.x
additional_train = train_dataset.additional
y_train = train_dataset.y
x_test = test_dataset.x
additional_test = test_dataset.additional
y_test = test_dataset.y

model.to('cpu')
pred, preds = model(x_test, additional_test)
y_pred_proba = model.predict_proba(x_test, additional_test).detach()
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
test_auc = metrics.roc_auc_score(y_test, y_pred_proba)
#create ROC curve
plt.plot(fpr,tpr, color="darkorange", label="ROC curve (area = %0.2f)" % test_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(loc="lower right")
plt.show()

# %%
model

# %%


# %%
class_weights = util.get_class_weights(torch.tensor(test_y_bin.values).view(-1))
model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run_geneset(genetic_data,
                                                                         test_y_bin,
                                                                        geneset_path='/mnt/disks/pancan/pnet/data/hallmark/c6.all.v2022.1.Hs.symbols.gmt',
                                                                         verbose=True,
                                                                         early_stopping=True,
                                                                         epochs=10,
                                                                         batch_size=10,
                                                                         loss_weight=class_weights,
                                                                         loss_fn=nn.BCEWithLogitsLoss(reduce=None),
                                                                         genes = canc_genes
                                                                        )

x_train = train_dataset.x
additional_train = train_dataset.additional
y_train = train_dataset.y
x_test = test_dataset.x
additional_test = test_dataset.additional
y_test = test_dataset.y

model.to('cpu')
pred, preds = model(x_test, additional_test)
y_pred_proba = model.predict_proba(x_test, additional_test).detach()
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
test_auc = metrics.roc_auc_score(y_test, y_pred_proba)
#create ROC curve
plt.plot(fpr,tpr, color="darkorange", label="ROC curve (area = %0.2f)" % test_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(loc="lower right")
plt.show()

# %%
model

# %%
import GenesetNetwork
gn = GenesetNetwork.GenesetNetwork(canc_genes, '/mnt/disks/pancan/pnet/data/hallmark/c6.all.v2022.1.Hs.symbols.gmt')

# %%
gn.pathway_encoding['ID'].nunique()

# %%
[l.shape for l in gn.gene_layers]

# %%
[l.shape for l in gn.pathway_layers]

# %%
gn.is_no_bugs()

# %%
gn.pathway_layers[0]

# %%
gn.genes2pathways[gn.genes2pathways['gene'] == 'CCND2']

# %%
gn.genes2pathways.index.value_counts()

# %%
gn.genes2pathways['gene']

# %%
pd.get_dummies(gn.genes2pathways['gene']).join(gn.genes2pathways['pathway']).groupby('pathway').sum().T

# %%


# %%
df = pd.DataFrame(index=['train_loss', 'test_loss'], data=[train_scores, test_scores]).transpose()
df['auc'] = 0.75

# %%
df

# %%
task = util.get_task(test_y_bin)
target = util.format_target(test_y_bin, task)
train_dataset, test_dataset = pnet_loader.generate_train_test(genetic_data, target)
reactome_network = ReactomeNetwork.ReactomeNetwork(train_dataset.get_genes())

# %%
model = Pnet.PNET_NN(reactome_network=reactome_network, task=task, nbr_gene_inputs=len(genetic_data),
                     loss_weight=class_weights, loss_fn=nn.BCEWithLogitsLoss(reduce=False))
train_loader, test_loader = pnet_loader.to_dataloader(train_dataset, test_dataset, 10)
model, train_scores, test_scores = Pnet.train(model, train_loader, test_loader, save_path='../results/model', epochs=10)

# %%
def deeppathwayVAE_mask(gene_list, n_level=5):
    network = ReactomeNetworkOG.ReactomeNetwork(gene_list, n_levels=n_level)
    return network

# %%
ReactomeNetwork = deeppathwayVAE_mask(gene_list, n_level=5)

# %%
ReactomeNetwork.masks[-2].shape

# %%
model.reactome_network.pathway_layers[0].shape

# %%
canc_genes = list(pd.read_csv('../../pnet_database/genes/cancer_genes.txt').values.reshape(-1))

# %%



