import os
import shutil
from pathlib import Path

from marl_factory_grid.utils.tools import ConfigExplainer


def init():
    print('Retrieving available options...')
    ce = ConfigExplainer()
    cwd = Path(os.getcwd())
    ce.save_all(cwd / 'full_config.yaml')
    template_path = Path(__file__).parent / 'modules' / '_template'
    print(f'Available config options saved to: {(cwd / "full_config.yaml").resolve()}')
    print('-----------------------------')
    print(f'Copying Templates....')
    shutil.copytree(template_path, cwd)
    print(f'Templates copied to {cwd}"/"{template_path.name}')
    print(':wave:')

