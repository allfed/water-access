# Water-Access
---


<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6866654.svg)](https://doi.org/10.5281/zenodo.6866654) -->
![Testing](https://github.com/allfed/water-access/actions/workflows/testing.yml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---
The Water Access model is a tool that simulates global access to freshwater sources by walking or cycling in the event of global catastrophic infrastructure loss (CIL). The output of the model can be used to identify areas at various resolutions (global, continental, country, district, and 5-arcminute zonal) which will be at highest risk of water insecurity in a catastrophe. This model can be useful for researchers, developers, and disaster response teams who are interested in understanding the potential impact of CIL on water access. The model is implemented in Python and its outputs have the potential to be integrated into other analysis and visualization tools (e.g., QGIS).

![Distributions plot](https://github.com/allfed/water-access/blob/main/results/access_distributions.png)

## Installation
To install the Water Access package, we recommend setting up a virtual environment using `mamba` (faster) or `conda`. This will ensure that the package and its dependencies are isolated from other projects on your machine, which can prevent conflicts and make it easier to manage your dependencies. Here are the steps to follow:

* Create a virtual environment using either conda by running the command `mamba env create -f environment.yml`. This will create an environment called "water-access". A virtual environment is like a separate Python environment, which you can think of as a separate "room" for your project to live in, it's own space which is isolated from the rest of the system, and it will have it's own set of packages and dependencies, that way you can work on different projects with different versions of packages without interfering with each other.

* Activate the environment by running `mamba activate water-access`. This command will make the virtual environment you just created the active one, so that when you run any python command or install any package, it will do it within the environment.

* Install the package by running `pip install -e .` in the main folder of the repository. This command will install the package you are currently in as a editable package, so that when you make changes to the package, you don't have to reinstall it again.

* The code is split between python files and Jupytper notebooks. If you want to run the Jupyter notebooks, you'll need to create a kernel for the environment. First, if using Anaconda, install the necessary tools by running `conda install -c anaconda ipykernel`. This command will install the necessary tools to create a kernel for the Jupyter notebook. A kernel is a component of Jupyter notebook that allows you to run your code. It communicates with the notebook web application and the notebook document format to execute code and display the results.

* Then, create the kernel by running `python -m ipykernel install --user --name=water-access`. This command will create a kernel with the name you specified "water-access" , which you can use to run the example notebook or play around with the model yourself.

* Alternatively, you may wish to run the notebook in an IDE such as Visual Studio Code (instructions [here](https://code.visualstudio.com/docs/datascience/jupyter-notebooks))

If you are using the kernel and it fails due an import error for the model package, you might have to rerun: `pip install -e .`.

If you encounter any issues, feel free to open an issue in the repository.

## How this model works in general

At its most simple, this notebook uses the Lankford walking model and Martin cycling model to simulate water access at different locations all across the globe.

A quick start is provided at the end of this README, along with step-by-step instructions to repeat our analysis in the `docs` folder.

## Getting the data

The data is stored using [Git Large File Storage (LFS)](https://git-lfs.com/) as some file sizes are quite large. You will need to set up LFS to be able to download these properly and run the models.

## Structure

├── `data` All input data\
│   ├── `GIS` Spatial data\
│   ├── `lookup tables` Parameters read into models\
│   ├── `original_data` Unmodified source data\
│   └── `processed` Data used and generated by models\
│       └── `semi-processed` Partially processed data\
├── `docs` Step-by-step guidance on re-running analysis\
├── `results` Tables and plots\
├── `scripts` Scripts to process data, explore modelling, run models, and generate results\
│   ├── `Data Manipulation Scripts` Scripts to pre- and post-process data\
├── `src` Source code\
└── `tests` Tests

## Quick Start
This section provides a basic way to run some of the model scripts and results, without getting into the more complex aspects of re-generating the processed datasets.


### Global model (use existing results)
To explore the models with pre-generated results, the following script can be used:
* `scripts/key_results.ipynb`: Generates key results (data and plots) used in publication.

### Re-run the global model with new assumptions (create new results)

If you want to re-run the model while inputting your own assumptions, the scripts below provide the simplest way to achieve this.
* `scripts/run_monte_carlo.ipynb`: Performs Monte Carlo simulations of the global model runs. A set of constants at the start of the file are used to define the 90% confidence intervals for the Monte Carlo parameters, and can be changed to the users preferences. Running the model is resource-intensive, and guidance for settings is detailed in the code. 
* `scripts/key_results.ipynb`: As before, this generates key results (data and plots). The results from the new model run can then be directly compared to those used in publication.
* `src/gis_global_module.py`: Alternatively, if you would like to run the global model once without Monte Carlo simulations, the main function in this file can be used.
