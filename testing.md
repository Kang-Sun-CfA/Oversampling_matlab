# Golden Test Data

## Overview

The data/test directory contains the inputs and expected outputs used to determine that everything works as expected when pulling in the latest version of the L2-L3 code.

## One-time Setup
<details>
    <summary>Click to expand</summary>

Create the conda environment. From the root directory of the github repo

```
conda env create -f environment.yml
```
</details>


## How to run

#TODO: ## Automate the running of these tests
#TODO: ## Generate and compare netcdf outputs

Run the following steps from the root directory of this github repo

1. Make sure you have the latest version of the test data

    ```
    dvc update
    ```

1. conda activate level3
1. Run the python

    ```
    python3 msat_l3.py -i data/test/input -o /tmp --start-time 2016-10-01 --end-time 2016-10-30 -w -100 -e -95 -n 35 -s 30 --grid-size 0.001
    ```

1. Make sure there are no unexpected errors or warnings.
1. Diff the outputs.

    ```
    diff /tmp/l3.png data/test/output/l3.png
    ```