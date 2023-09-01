from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(name='Marl-Factory-Grid',
      version='0.1.2',
      description='A framework to research MARL agents in various setings.',
      author='Steffen Illium',
      author_email='steffen.illium@ifi.lmu.de',
      url='https://github.com/illiumst/marl-factory-grid/import',
      license='MIT',
      keywords=[
            'artificial intelligence',
            'pytorch',
            'multiagent reinforcement learning',
            'simulation',
            'emergence',
            'gymnasium',
            'environment',
            'deepdiff',
            'natsort',

      ],
      classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.11',
      ],
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(exclude=['examples']),
      include_package_data=True,
      install_requires=['numpy', 'pygame>=2.0', 'numba>=0.56', 'gymnasium>=0.26', 'seaborn', 'pandas',
                        'pyyaml', 'networkx', 'torch', 'tqdm']
      )
