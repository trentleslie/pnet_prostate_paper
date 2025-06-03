# %% [markdown]
# # P-Net Example Notebook: SKCM tumor purity prediction

# %% [markdown]
# Imports:

# %%
from pnet import pnet_loader, Pnet
from util import util, sankey_diag

import torch
import seaborn as sns
import pandas as pd
import numpy as np
import random
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

%load_ext autoreload
%autoreload 2

# %% [markdown]
# ### Load Input

# %% [markdown]
# The input in this case is a count matrix from rna expression and copy number amplifications on a gene level of the SKCM dataset from TCGA. We use a custom function for some formatting, but pandas would work fine too. 
# 
# The data is then put into a dictionary shape required for pnet, this is especially useful when using multiple data modalities. 

# %%
datapath='/mnt/disks/pancan/data/'
rna, cna, tumor_type = util.load_tcga_dataset(datapath+'skcm_tcga_pan_can_atlas_2018')
genetic_data = {'rna': rna, 'cna': cna}                         # Bring input into dictionary shape

# %% [markdown]
# ### Load Target

# %% [markdown]
# In this example notebook we use tumor purity as a target, which serves as a good example for a regression task with gene level data. We use the tumor purity called by  Taylor et al. 2018, available here https://gdc.cancer.gov/about-data/publications/pancanatlas

# %%
purity_TCGA = pd.read_csv(datapath+'TCGA_mastercalls.abs_tables_JSedit.fixed.txt', delimiter='\t').set_index('array')
purity = pd.DataFrame(purity_TCGA.join(rna, how='inner')['purity'])

# %% [markdown]
# We have the tumor purity as a continuous number in [0,1] and can use this for a regression task

# %%
purity.head()

# %% [markdown]
# ## Run pnet

# %% [markdown]
# We limit the number of genes used for prediction to some subset of informative cancer genes

# %%
canc_genes = list(pd.read_csv('../../pnet_database/genes/cancer_genes_demo.txt'))
canc_genes[:5]

# %% [markdown]
# The Pnet model can then be trained by simply calling Pnet.run() with the input dictionairy and the target dataframe. Additionally, here we use the cancer gene list. Many more options are available to specific training.

# %%
model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(genetic_data,
                                                                         purity,
                                                                         gene_set=canc_genes)

# %% [markdown]
# The Pnet model automatically generates a train and test dataset from the input. Here we can use these for illustration of the results on the test dataset:

# %%
x_train = train_dataset.x
additional_train = train_dataset.additional
y_train = train_dataset.y
x_test = test_dataset.x
additional_test = test_dataset.additional
y_test = test_dataset.y

# %%
model.to('cpu')

y_pred = model.predict(test_dataset.x, test_dataset.additional).detach()

df = pd.DataFrame(index=test_dataset.input_df.index)
df['y_test'] = test_dataset.y
df['y_pred'] = y_pred

sns.regplot(data=df, x='y_test', y='y_pred', color='#41B6E6')
correlation_coefficient = round(df['y_test'].corr(df['y_pred']), 2)
plt.text(0.95, 0.05, f'Correlation: {correlation_coefficient}', ha='right', va='center', transform=plt.gca().transAxes)
plt.plot(y_test, y_test, color='#FFA300', linestyle='--', label='Diagonal Line')
sns.despine()
plt.show()

# %% [markdown]
# For a quick look at the importance scores we can run model.interpret with the test dataset. We recommend however, to use multiple folds of cross validation before inspecting feature and pathway importances, since individual runs can drastically differ and only stable features across runs can be considered important.

# %%
gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores = model.interpret(test_dataset)
layer_list = [gene_feature_importances, additional_feature_importances, gene_importances] + layer_importance_scores
layer_list_names = ['gene_feature', 'additional_feature', 'gene'] + [f'layer_{i}' for i in range(5)]
layer_list_dict = dict(zip(layer_list_names, layer_list))

# %% [markdown]
# To visualize feature importances, one can generate a Sankey Diagram across all layers of the model:

# %%
savepath = '/mnt/disks/pancan/pnet/figures/skcm_purity_sankey.html'
sk = sankey_diag.SankeyDiag(layer_list_dict, runs=1)
fig = sk.get_sankey_diag(savepath)

# %% [markdown]
# ### External Validation

# %% [markdown]
# #### Data Loading

# %% [markdown]
# To show generalizability of Pnet we show the performance on an external validation dataset. The Liu 2019 cohort (https://doi.org/10.1038/s41591-019-0654-5) was used. Where purity has also been identified with the Absolute algorithm.

# %%
rna_ext_val = pd.read_csv('/mnt/disks/pancan/data/mel_dfci_2019/data_RNA_Seq_expression_tpm_all_sample_Zscores.txt',
                          delimiter='\t').set_index('Hugo_Symbol').T.drop('Entrez_Gene_Id').dropna(axis=1)
cna_ext_val = pd.read_csv('/mnt/disks/pancan/data/mel_dfci_2019/data_CNA.txt',
                          delimiter='\t').set_index('Hugo_Symbol').T.dropna(axis=1)

ext_val = pd.read_csv('/mnt/disks/pancan/data/mel_dfci_2019/data_clinical_sample.txt',
                             delimiter='\t').set_index('Sample Identifier').iloc[4:]

ext_val_target = ext_val['Purity'].astype('float')

inds = pnet_loader.get_indicies(genetic_data = {'rna': rna_ext_val, 'cna': cna_ext_val},
                                target=ext_val_target)

ext_val_dataset = pnet_loader.PnetDataset(genetic_data = {'rna': rna_ext_val, 'cna': cna_ext_val},
                                          target=ext_val_target, indicies=inds, gene_set=canc_genes)

x_val = ext_val_dataset.x
additional_val = ext_val_dataset.additional
y_val = ext_val_dataset.y

# %% [markdown]
# #### Predict Validation Samples

# %%
y_pred = model.predict(x_val, additional_val).detach()

# %% [markdown]
# #### Plot Validation Samples

# %%
df = pd.DataFrame(index=ext_val_dataset.input_df.index)
df['y_test'] = y_val
df['y_pred'] = y_pred

sns.regplot(data=df, x='y_test', y='y_pred', color='#41B6E6')
correlation_coefficient = round(df['y_test'].corr(df['y_pred']), 2)
plt.text(0.95, 0.05, f'Correlation: {correlation_coefficient}', ha='right', va='center', transform=plt.gca().transAxes)
plt.plot(y_test, y_test, color='#FFA300', linestyle='--', label='Diagonal Line')
sns.despine()
plt.show()

# %% [markdown]
# We can see that even in a independent dataset Pnet is generally predicting purity well. It is overestimating the purity in the validation set slightly, especially for lower purity samples. Maybe the performance could be improved by using a customary loss function for the problem at hand.

# %% [markdown]
# ## Use custom loss function

# %% [markdown]
# We define a weighted MSE loss that penalizes bad predictions in extreme samples more. Bad predictions in a sample with very high or very low purity would be more influencial to the model than in average purity samples.

# %%
import torch.nn as nn
import torch.nn.functional as F

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, input, target):
        # Calculate the absolute distance of the true values from 0.5
        distance_from_center = torch.abs(target - 0.5)
        # Scale weights as needed; further from 0.5 gets higher weight
        # This example linearly increases the weight as the distance from 0.5 increases
        weights = 1 + distance_from_center 
        # Calculate weighted MSE
        se = (input - target) ** 2
        weighted_se = weights * se
        loss = weighted_se.mean()
        return loss


# %% [markdown]
# The custom loss function easily integrates into the model training.

# %%
model, train_scores, test_scores, train_dataset, test_dataset = Pnet.run(genetic_data,
                                                                         purity,
                                                                         gene_set=canc_genes,
                                                                         loss_fn = WeightedMSELoss())

# %%
model.to('cpu')

y_pred = model.predict(x_test, additional_test).detach()

df = pd.DataFrame(index=test_dataset.input_df.index)
df['y_test'] = y_test
df['y_pred'] = y_pred

sns.regplot(data=df, x='y_test', y='y_pred', color='#41B6E6')
correlation_coefficient = round(df['y_test'].corr(df['y_pred']), 2)
plt.text(0.95, 0.05, f'Correlation: {correlation_coefficient}', ha='right', va='center', transform=plt.gca().transAxes)
plt.plot(y_test, y_test, color='#FFA300', linestyle='--', label='Diagonal Line')
sns.despine()
plt.show()

# %% [markdown]
# Performance in the test dataset is already much higher. Especially in extreme samples.

# %%
y_pred = model.predict(x_val, additional_val).detach()
df = pd.DataFrame(index=ext_val_dataset.input_df.index)
df['y_test'] = y_val
df['y_pred'] = y_pred

sns.regplot(data=df, x='y_test', y='y_pred', color='#41B6E6')
correlation_coefficient = round(df['y_test'].corr(df['y_pred']), 2)
plt.text(0.95, 0.05, f'Correlation: {correlation_coefficient}', ha='right', va='center', transform=plt.gca().transAxes)
plt.plot(y_test, y_test, color='#FFA300', linestyle='--', label='Diagonal Line')
sns.despine()
plt.show()


