This repository contains supplementary source code and analysis of the [master thesis](https://github.com/uvNikita/master-thesis) project.
[vagga](https://github.com/tailhook/vagga) is used to create a reproducible environment. Therefore, in order to start [Jupyter](http://jupyter.org/) with the correct environment run:

```
$ vagga jupyter
```
This command will build all necessary libraries for training, testing, and analysis of used neural networks.
Since dataset employed in this research was provided by Norwegian News Agency ([NTB](http://www.ntb.no)), to reuse code for training networks on other datasets, it has to be slightly adjusted.

The [notebooks](https://github.com/uvNikita/master-thesis-src/tree/master/notebooks) folder contains [Jupyter](http://jupyter.org/) notebooks used for all training, testing and analysis parts.

* [metadata.ipynb](https://github.com/uvNikita/master-thesis-src/blob/master/notebooks/metadata.ipynb) contain the whole pipeline of metadata parsing, translation and restructuring. This file is the most dataset-specific. However, it can serve as an example of how the final structure should look like to work with other parts of the project.
* [database.ipynb](https://github.com/uvNikita/master-thesis-src/blob/master/notebooks/database.ipynb) contain general analysis of the used dataset.
* [training])(https://github.com/uvNikita/master-thesis-src/blob/master/notebooks/training) folder has several notebooks. Each of them represents one training process for each combination of category and network selection.
* [testing](https://github.com/uvNikita/master-thesis-src/blob/master/notebooks/testing) folder contain one testing notebook where all trained networks are analyzed and compared.
* [ntb](https://github.com/uvNikita/master-thesis-src/blob/master/notebooks/ntb) folder contain additional library tools that were written used in the process.
