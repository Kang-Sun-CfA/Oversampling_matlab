# RRNES
This folder contains basic Python code to estimate air basin-scale emissions using
 the method developed by [Sun et al. (2021)](https://acp.copernicus.org/preprints/acp-2021-268/).
 ## Installation
 It is suggested to use the conda environment yml file given at the [top level of this repository](https://github.com/Kang-Sun-CfA/Oversampling_matlab).
 ## Files
 ### `rrnes.py`
 The main package containing routines to fit emission rate and chemical lifetime from the column-wind speed relationships.
 ### `rrnes_level3.py`
 A script wrapping around functions in [`popy.py`](https://github.com/Kang-Sun-CfA/Oversampling_matlab/blob/master/popy.py). It averages column amount within a basin boundary over different wind speed intervals. The results will be level 3 files containing the column-wind speed relationship data.
### `rrnes_level4.py` 
A script calling functions in [`rrnes.py`](https://github.com/Kang-Sun-CfA/Oversampling_matlab/blob/master/RRNES/rrnes.py) and generating emission/lifetime (level 4 data) from averaged satellite column amount (level 3 data).
