# MSat Data Platform

This repository is the Data Platform orchestration system, built on top of 
Flyte.

Setup is simple - just run `python -m pip install -r requirements.txt` from the
`orchestration` folder.

Internally, this installs any necessary packages and then installs the 
`orchestration` folder in editable mode. See 
[pip install](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs) 
for further info. Alternatively, you can achieve the same thing manually by 
running `pip install --editable .`

From there, you can run each task or workflow file directly through python 
`python3 workflow/test_all.py`

or by spinning up a Flyte instance. 
