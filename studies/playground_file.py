import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


dfs = []
for name in ['l2snac', 'iac', 'snac', 'seac']:
    for c in range(5):
        try:
            study_root = Path(__file__).parent / name / f'{name}#{c}'
            df = pd.read_csv(study_root / 'results.csv', index_col=False)
            df.reward = df.reward.rolling(100).mean()
            df['method'] = name.upper()
            dfs.append(df)
        except Exception as e:
            pass

df = pd.concat(dfs).reset_index()
sns.lineplot(data=df, x='episode', y='reward', hue='method', palette='husl', ci='sd', linewidth=1.5)
plt.savefig('study.png')
print('saved image')