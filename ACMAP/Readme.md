# ACMAP
This folder contains basic Python code to derive surface concentrations using an observational data-driven approach.
 ## Installation
 It is suggested to use the conda environment yml file given at the [top level of this repository](https://github.com/Kang-Sun-CfA/Oversampling_matlab).
 ## Files
 ### `aircraft_profiles.py`
 A package containing routines to deal with DISCOVER-AQ aircraft spiral profiles. The main objective is to investigate the similarity in the vertical distribution of reactive species (e.g., nitrogen oxides and formaldehyde) after the vertical coordinate (pressure) and mixing ratios are non-dimensionalized. The data files originates from the [NASA DISCOVER-AQ archive](https://www-air.larc.nasa.gov/missions/discover-aq/discover-aq.html), but currently `aircraft_profiles.py` can only load processed `.mat` files.
