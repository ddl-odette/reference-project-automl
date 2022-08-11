*Disclaimer - Domino Reference Projects are starter kits built by Domino researchers. They are not officially supported by Domino. Once loaded, they are yours to use or modify as you see fit. We hope they will be a beneficial tool on your journey! 

## Welcome to the Domino Reference Project for...

# AutoML

![image](https://github.com/hibayesian/awesome-automl-papers/raw/master/resources/banner.png)
-image from Mark Lin, https://github.com/hibayesian/awesome-automl-papers

There are many proprietary and open source AutoML tools available. This project dives into a few of them with the goal of getting you started with hands-on AutoML in Domino. 

## Project Contents

* [auto-sklearn.ipynb](./view/auto-sklearn.ipynb), a how-to notebook
* [tpot.ipynb](./view/tpot.ipynb), a how-to notebook
* [mlbox.ipynb](./view/MLBox.ipynb), a how-to notebook
* AutoML Launcher, associated [configuration file](./view/launcher_config.txt), and [launcher set up doc](./view/launcher_setup.md)
* [automl.py](./view/automl.py), the code behind the Launcher
* sample data files

## Suggested Actions

* Browse the notebooks. They are writen as tutorials on how to use the various automl packages.
* Copy or Fork this project to run the notebooks on your own. 
* Follow the instructions in the [launcher set up doc](./view/launcher_setup.md) to create your own AutoML Launcher.

## Reference Material

Check out [this great site](https://github.com/hibayesian/awesome-automl-papers) by Mark Lin for an overview of AutoML.

Check out this [spreadsheet](https://docs.google.com/spreadsheets/d/1KVtbJfBcjnh_0YIgfLyfROxDHtcw8QOafjTicjyiUxo/edit#gid=1849753649) from Paco Nathan listing the proprietary and open source AutoML tools.

Each of the jupyter notebooks listed above has additional links to reference material.

## Prerequisites

This project uses standard python libraries and any base Domino image should work well. The last test was done on standard-environment:ubuntu18-py3.8-r4.1-domino5.1. The additional Python libraries needed are shap, lime, and pycebox. You can simply install them in the cell provided when running the notebook interactively. Alternatively, you can add them to a custom compute environment by appending the following lines to the standard-environment:ubuntu18-py3.8-r4.1-domino5.1 dockerfile:

RUN echo "ubuntu    ALL=NOPASSWD: ALL" >> /etc/sudoers
RUN pip install --upgrade pip
RUN pip install pycebox \
                lime \
                shap
There are several additional R libraries needed to run rtemis. This library changes frequently, sometimes breaking dependencies, so we do not advise building a compute environment for the current dependencies. See the R scipt included in this project for details.