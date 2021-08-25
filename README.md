# MSat Data Platform

This repository is the Data Platform orchestration system, built on top of 
Flyte.

Setup is simple - just run `python -m pip install -r requirements_dev.txt` from the
`orchestration` folder.

Internally, this installs any necessary packages and then installs the 
`orchestration` folder in editable mode. See 
[pip install](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs) 
for further info. Alternatively, you can achieve the same thing manually by 
running `pip install --editable .`

From there, you can run each task or workflow file directly through python 
`python3 workflow/test_all.py`

or by spinning up a Flyte instance.

## Data

Test data associated with sci-level3 is tracked by dvc. To download please run

```bash
dvc pull
```

## Cartopy Dependency Installation 

Cartopy is needed for L3 visualizations. It depends on GEOS and Proj being installed on the machine and accesible in the environment PATH and LD_LIBRARY_PATH

```
# https://trac.osgeo.org/geos
brew install geos
# OR
# apt install libgeos-dev

# install Proj
# https://proj.org/install.html#install
# needed to install v7 because proj_api.h was missing from version 8
brew install proj@7
# OR
# apt-get install libproj-dev

export PATH=$PATH:/usr/local/Cellar/proj@7/7.2.1/bin

# https://stackoverflow.com/questions/18783390/python-pip-specify-a-library-directory-and-an-include-directory
pip install --global-option=build_ext --global-option="-I/usr/local/Cellar/proj@7/7.2.1/include" --global-option="-L/usr/local/Cellar/proj@7/7.2.1/lib" cartopy==0.18.0
```