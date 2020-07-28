import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


dataset_label = 'dataset/d2h'
intdim_label = 'intrinsic dim_L1'
var_path = 'data/extra/variance.csv'
uci_datasets = ['adult','annealing','audit','autism','bank','bankrupt','biodegrade','blood-transfusion','cancer','car','cardiotocography','cervical-cancer','climate-sim','contraceptive','covtype','credit-approval','credit-default','crowdsource','diabetic','drug-consumption','electric-stable','gamma','hand','hepmass','htru2','image','kddcup','liver','mushroom','optdigits','pendigits','phishing','satellite','sensorless-drive','shop-intention','shuttle','waveform']

df = pd.read_csv(var_path)
df = df[[dataset_label, intdim_label]]
df.dropna(inplace=True)

def get_source(row):
    if (row[dataset_label] in uci_datasets):
        return 'uci'
    else:
        return 'se'

df['source'] = df.apply(lambda row : get_source(row), axis=1)

df = df.sort_values(intdim_label, ignore_index=True)

print(df)
n_rows = df.shape[0]
x_pos = np.arange(n_rows)
print('n_rows = ' + str(n_rows))
print('xpos = ' + str(x_pos))

# Constructing the bar graph
uci_color = 'red'
se_color = 'blue'
plt.rcParams.update({'figure.autolayout': True})
c_dict = {'uci' : uci_color, 'se': se_color}
sources = df['source'].values.tolist()
colors = [c_dict.get(x) for x in sources]
print('colors = ' + str(colors))
x_labels = df[dataset_label].to_numpy()
fig, ax = plt.subplots()
ax.bar(x_labels, df[intdim_label].to_numpy(), color=colors)
ax.set(xlabel = 'Dataset', ylabel = 'Intrinsic dimensionality', title = 'Intrinsic dimensionalities for datasets')
xtick_labels = ax.get_xticklabels()
plt.setp(xtick_labels, rotation='vertical')
uci_patch = mpatches.Patch(color = uci_color, label = 'UCI datasets')
se_patch = mpatches.Patch(color = se_color, label = 'SE datasets')
plt.legend(handles=[uci_patch, se_patch], loc='upper left')
#ax.legend([uci_patch, se_patch], ['uci', 'se'])
#plt.bar(x_pos, df[intdim_label], color=colors)
#plt.xlabel('dataset')
#plt.ylabel('intrinsic dimensionality')
#plt.title('Intrinsic dimensionalities for datasets')
#plt.xticks(x_pos, df[dataset_label], rotation='vertical')
plt.show()

print(df)

