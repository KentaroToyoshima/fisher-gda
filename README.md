# Min-Max GDA-Fisher

Min-Max GDA-Fisher is a framework for evaluating the performance of algorithms across different market types.

## Directory Structure

The directory structure for the project is as follows:

- `main.py`: This file contains the main Python script that runs the framework.

- `utils.py`: This file contains utility functions and helper scripts used by the framework.

- `scripts.sh`: This file contains shell scripts used for various tasks in the framework.

- `requirements.txt`: This file lists the Python libraries required by the framework.

- `README.md`: This file contains the documentation and instructions for using the Min-Max GDA-Fisher framework.

- `fisherMinmax.py`: This file contains implementations of various LGDA algorithms.

- `fisher_env.yml`: This file is an environment configuration file for setting up the required dependencies.

- `consumerUtility.py`: This file contains the implementation of the consumer utility functions.

- `assemble_plot.py`: This file contains the code for assembling and plotting the results of the experiments.

- `results/`: This directory is used to store the results of the experiments.

## Requirements

### Programming Languages
- Python 3.7

### Python Libraries
To install python requirements, run the following command:
```
pip install -r requirements.txt
```


## Command Line Arguments

The command line arguments are as follows:

- `--market_types -mt`: The types of markets to be run. Options are 'linear', 'cd', 'leontief'. The default is all types.
- `--num_experiments -e`: The number of experiments to be run. The default is 5.
- `--num_buyers -b`: The number of buyers in the market. The default is 5.
- `--num_goods -g`: The number of goods in the market. The default is 8.
- `--learning_rate_linear -li`: The learning rate for the linear market. The default is [0.01, 0.01].
- `--learning_rate_cd -cd`: The learning rate for the CD market. The default is [0.01, 0.01].
- `--learning_rate_leontief -Le`: The learning rate for the Leontief market. The default is [0.01, 0.01].
- `--mutation_rate -mu`: The mutation rate for each market type. The default is [1, 1, 1].
- `--num_iters -i`: The number of iterations. The default is 1000.
- `--update_freq -u`: The frequency of updates. The default is 0.
- `--arch -a`: The algorithm to be used. Options are 'alg2', 'm-alg2', 'alg4'. The default is 'alg4'.

## Usage Example

Here is an example of how to run the framework:

```
python main.py -mt linear cd leontief -e 10 -b 5 -g 8 -li 0.01 0.01 -cd 0.01 0.01 -Le 0.001 0.001 -mu 0.1 0.1 1 -i 1000 -u 0 -a alg4
```

This command runs 10 experiments for all market types 'linear', 'cd', 'leontief'. Each market consists of 5 buyers and 8 goods. Also, the learning rate for linear, CD, and Leontief markets are all [0.01, 0.01], with mutation rates of 0.1, 2, and 5 respectively. Each experiment runs 1000 iterations and uses the 'alg4' algorithm.
