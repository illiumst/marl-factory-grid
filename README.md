# Emergence in Multi-Agent Systems: A Safety Perspective

## Setup

1. Set up a virtualenv with python 3.10 or higher. You can use pyvenv or conda for this.
2. Run ```pip install -r requirements.txt``` to get requirements.
3. In case there is no ```study_out/``` folder in the root directory, create one.

## Rerunning the Experiments 

The respective experiments from our paper can be reenacted in ```main.py```.
Just select the function representing the part of our experiments you want to rerun and 
execute it via the ```__main__``` function.

## Further Remarks
1. We use config files located in the ```marl_factory_grid/environment/configs``` and the 
```marl_factory_grid/algorithms/rl``` folders to configure the environments and the RL
algorithm for our experiments, respectively. You don't need to change anything to rerun the 
experiments, but we provided some additional comments in the configs for an overall better
understanding of the functionalities.
2. Instead of collecting coins in the coin-quadrant environment our original implementation 
works with the premise of cleaning piles of dirt, thus it is named ```dirt_quadrant``` in the code instead. 
Note that this difference is only visual and does not change the underlying semantics of the environment.
3. The code for the cost contortion for preventing the emergent behavior of the TSP agents can
be found in ```marl_factory_grid/algorithms/tsp/contortions.py```.
4. The functionalities that drive the emergence prevention mechanisms for the RL agents is mainly 
located in the utility functions ```get_ordered_dirt_piles (line 91)``` (for solving the emergence in the 
coin-quadrant environment) and ```distribute_indices (line 165)``` (mechanism for two_doors), that are part of
```marl_factory_grid/algorithms/rl/utils.py```



