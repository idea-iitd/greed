# GREED: A Neural Framework for Learning Graph Distance Functions

This repository contains the official reference implementation for the paper ["GREED: A Neural Framework for Learning Graph Distance Functions"](https://openreview.net/pdf?id=3LBxVcnsEkV) accepted at NeurIPS 2022. `neuro` contains our implementation of the neural models presented in the paper along with supporting code for experiments. `pyged` contains our python wrapper over [GEDLIB](https://github.com/dbblumenthal/gedlib), which can be used to compute SED/GED values and graph alignments using non-neural techniques.

## Data

The data and trained models can be downloaded from this [Google Drive link](https://drive.google.com/file/d/1bRf6isnbfIrDc7V8xStlwEFX1ZtIMBEB/view?usp=sharing). Please see the README contained therein for further details.

## Experiments

The Jupyter notebooks for the experiments in the paper can be found at the sister repository [greed-expts](https://github.com/rishabh-ranjan/greed-expts).

## Installation

We recommend using a `conda` environment for installation.

1. Install _Python_, _Jupyter_, _PyTorch_ and _PyTorch Geometric_. The code has been tested to work with _Python 3.6.13_, _PyTorch 1.8.0_ and _PyTorch Geometric 1.6.3_, but later versions are also expected to run smoothly.

2. Install _pyged_:

	2.1. Install [GEDLIB](https://dbblumenthal.github.io/gedlib/) at `pyged/ext/gedlib` as a header-only library (see Section 4.1 in the docs).

	2.2. Install [Gurobi 9.1.1](https://support.gurobi.com/hc/en-us/articles/360054352391-Gurobi-9-1-1-released) at `pyged/ext/gurobi911`. Later versions can be used with suitable naming changes. _Gurobi_ requires a licence. Free academic licenses are available. _Gurobi_ is required for ground truth SED computation. Alternatively, one could use one of the non-MIP methods available in _GEDLIB_ or use the generated data provided by us. To build without _Gurobi_, uncomment `#define GUROBI` in `pyged/src/pyged.cpp`.

	2.3. Install [PyBind11](https://pybind11.readthedocs.io/en/stable/installing.html#include-with-conda-forge).

	2.4. Build _pyged_:
	```
	mkdir pyged/build
	cd pyged/build
	cmake ..
	make
	```
	This will create a Python module for _pyged_ in `pyged/lib`.

## Usage

Check out the experiment notebooks at [greed-expts](https://github.com/rishabh-ranjan/greed-expts) for example usage. The notebooks contain code for training, testing, visualization, etc.

## Contact

If you face any difficulties in using this repo feel free to raise a GitHub issue (recommended) or reach out via email at rishabhranjan0207@gmail.com. I am unable to respond to queries sent to rishabh.ranjan.cs118@cse.iitd.ac.in in a timely manner.

## Citation

```
@inproceedings{ranjan&al22,
  author = {Ranjan, Rishabh and Grover, Siddharth and Medya, Sourav and Chakaravarthy, Venkatesan and Sabharwal, Yogish and Ranu, Sayan},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {GREED: A Neural Framework for Learning Graph Distance Functions},
  booktitle = {Advances in Neural Information Processing Systems 36: Annual Conference
               on Neural Information Processing Systems 2022, NeurIPS 2022, November 29-Decemer 1, 2022},
  year = {2022},
}
```

