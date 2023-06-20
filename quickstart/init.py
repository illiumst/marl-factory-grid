import os
import shutil
from pathlib import Path

from mfg_package.utils.tools import ConfigExplainer

if __name__ == '__main__':
    print('Retrieving available options...')
    ce = ConfigExplainer()
    cwd = Path(os.getcwd())
    ce.save_all(cwd / 'full_config.yaml')
    template_path = Path(__file__) / 'mfg_package' / 'modules' / '_template'
    shutil.copytree(template_path, cwd)
    print()
