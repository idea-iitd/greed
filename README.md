# A Neural Framework for Learning Graph and Subgraph Similarity Measures

This repository contains the reference implementation for the paper "A Neural Framework for Learning Graph and Subgraph Similarity Measures". `neuro` contains our implementation of the neural models presented in the paper along with supporting code for experiments. `pyged` contains our python wrapper over [GEDLIB](https://github.com/dbblumenthal/gedlib), which can be used to compute SED/GED values and graph alignments using non-neural techniques.

## Data

The data and pre-trained models can be downloaded from this [Google Drive link](https://drive.google.com/file/d/11cCqmr0WEqfSoxubxvbIP4APBGTAwHrf/view?usp=sharing). Please see the README contained therein for further details.

## Experiments

The Jupyter notebooks for the experiments in the paper can be found at the sister repository [NeuroSED-Expts](https://github.com/rishabh-ranjan/NeuroSED-Expts).

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

Check out the experiment notebooks for example usage. Scripts will be added soon. Meanwhile, it should be easy enough to adapt the code from the notebooks for training, testing, visualization, etc.

