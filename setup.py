from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(name='Marl-Neon-Grid',
      version='0.1.4.4',
      description='A collection of MARL gridworlds to study coordination and cooperation.',
      author='Robert Müller',
      author_email='steffen.illium@ifi.lmu.de',
      url='https://github.com/romue404/marl-neon-grid',
      license='MIT',
      keywords=[
            'artificial intelligence',
            'pytorch',
            'multiagent reinforcement learning',
            'simulation'
      ],
      classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.6',
      ],
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(exclude=['examples']),
      include_package_data=True,
      install_requires=['numpy', 'pygame>=2.0', 'numba>=0.56', 'gymnasium>=0.26', 'seaborn', 'pandas']
      )