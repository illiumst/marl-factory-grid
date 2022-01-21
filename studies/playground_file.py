import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

study_root = Path(__file__).parent / 'entropy_study'
names_all = ['basic_gru', 'layernorm_gru', 'spectralnorm_gru', 'nonorm_gru']
names_only_1 = ['L2OnlyAh_gru', 'L2OnlyChAh_gru', 'L2OnlyMix_gru', 'basic_gru']
names_only_2 = ['L2NoCh_gru', 'L2NoAh_gru', 'nomix_gru', 'basic_gru']

names = names_only_2
#names = ['nonorm_gru']
# /Users/romue/PycharmProjects/EDYS/studies/normalization_study/basic_gru#3
csvs = []
for name in ['basic_gru', 'nonorm_gru', 'spectralnorm_gru']:
    for run in range(0, 1):
        try:
            df = pd.read_csv(study_root / f'{name}#{run}' / 'results.csv')
            df = df[df.agent == 'sum']
            df = df.groupby(['checkpoint', 'run']).mean().reset_index()
            df['method'] = name
            df['run_'] = run

            df.reward = df.reward.rolling(15).mean()
            csvs.append(df)
        except Exception as e:
            print(f'skipped {run}\t {name}')

csvs = pd.concat(csvs).rename(columns={"checkpoint": "steps*2e3", "B": "c"})
sns.lineplot(data=csvs, x='steps*2e3', y='reward', hue='method', palette='husl', ci='sd', linewidth=1.8)
plt.savefig('entropy.png')