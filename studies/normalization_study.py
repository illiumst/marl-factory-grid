from algorithms.utils import Checkpointer
from pathlib import Path
from algorithms.utils import load_yaml_file, add_env_props, instantiate_class, load_class
#from algorithms.marl import LoopSNAC, LoopIAC, LoopSEAC



for i in range(0, 5):
    for name in ['snac', 'mappo', 'iac', 'seac']:
        study_root = Path(__file__).parent / name
        cfg = load_yaml_file(study_root / f'{name}.yaml')
        add_env_props(cfg)

        env = instantiate_class(cfg['env'])
        net = instantiate_class(cfg['agent'])
        max_steps = cfg['algorithm']['max_steps']
        n_steps = cfg['algorithm']['n_steps']

        checkpointer = Checkpointer(f'{name}#{i}', study_root, cfg, max_steps, 50)

        loop = load_class(cfg['method'])(cfg)
        df = loop.train_loop(checkpointer)

