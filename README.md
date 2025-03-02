# AdaptiveHGPMPC_ECC25

## Instructions for setup:
- Clone the repository to your local machine.
- Install the requirements with pip install -r requirements.txt
- Run the setup file with ```pip install -e .``` in the project *root* directory.
- Update ```consts/vars.py```'s data_dir variable with the path to the cloned ```data_dir``` directory as this is necessary for some experiments.
- You should now be able to run the jupyter notebooks. In case of any issues, print sys.path in the notebook to ensure the src directory can be found.

## Notebooks:

### ECC25_adapmap_results:

This notebook visualizes the results from the adaptive mapping section of the paper. 

The first section demonstrates our proposed adaptive likelihood-prior trade-off scheme on a toy example. 

The rest of the notebook then demonstrates how the proposed approach also transfers well to the closed-loop control task along with an ablation example of the adaptation speed on control performance.