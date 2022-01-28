import pandas as pd
from algorithms.marl import LoopSNAC, LoopIAC, LoopSEAC
from pathlib import Path
from algorithms.utils import load_yaml_file
from tqdm import trange
study = 'example_config#0'
#study_root = Path(__file__).parent / study
study_root = Path('/Users/romue/PycharmProjects/EDYS/algorithms/marl/')

#['L2NoAh_gru', 'L2NoCh_gru', 'nomix_gru']:
render = True
eval_eps = 3
for run in range(0, 5):
    for name in ['example_config']:#['L2OnlyAh_gru', 'L2OnlyChAh_gru', 'L2OnlyMix_gru']: #['layernorm_gru', 'basic_gru', 'nonorm_gru', 'spectralnorm_gru']:
        cfg = load_yaml_file(study_root / study / 'config.yaml')
        #p_root = Path(study_root / study / f'{name}#{run}')
        dfs = []
        for i in trange(500):
            path = study_root / study / f'checkpoint_{161}'
            print(path)

            snac = LoopSEAC(cfg)
            snac.load_state_dict(path)
            snac.eval()

            df = snac.eval_loop(render=render, n_episodes=eval_eps)
            df['checkpoint'] = i
            dfs.append(df)

        results = pd.concat(dfs)
        results['run'] = run
        results.to_csv(p_root / 'results.csv', index=False)

#sns.lineplot(data=results, x='checkpoint', y='reward', hue='agent', palette='husl')

#plt.savefig(f'{experiment_name}.png')