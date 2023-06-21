import os
import shutil
from pathlib import Path

from marl_factory_grid.utils.tools import ConfigExplainer

if __name__ == '__main__':
    print('Retrieving available options...')
    ce = ConfigExplainer()
    cwd = Path(os.getcwd())
    ce.save_all(cwd / 'full_config.yaml')
    template_path = Path(__file__) / 'marl_factory_grid' / 'modules' / '_template'
    shutil.copytree(template_path, cwd)
    print()
