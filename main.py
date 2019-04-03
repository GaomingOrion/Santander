import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('./data/train.csv', header=0)
test = pd.read_csv('./data/test.csv', header=0)

target_mask = train['target'] == 1
non_target_mask = train['target'] == 0
y = train.target.values

cols = [i for i in test.columns if "var" in i]
print(train.shape)

a = train.apply(lambda x: x.unique().shape[0])
# var_68 451

ks_pvalue = []
for col in train.columns[2:]:
    statistic, pvalue = ks_2samp(train.loc[non_target_mask, col], train.loc[target_mask, col])
    ks_pvalue.append(pvalue)
    # fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    # sns.kdeplot(train.loc[non_target_mask, col], ax=ax, label='Target == 0')
    # sns.kdeplot(train.loc[target_mask, col], ax=ax, label='Target == 1')
    #
    # ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))
    # plt.show()
ks_pvalue = np.array(ks_pvalue)
chosen = train.columns[2:][np.argwhere(ks_pvalue<0.05).reshape(-1)]