Adversarial Examples
====================
This is the code for the technical report *End-to-End Adversarial White Box Attacks on 
Music Instrument Classification* (Institute for Computational Perception, JKU Linz). To view
our supplementary material, including listening examples, 
look at our [Github page](https://prkat.github.io/adversarial_instrument_classification/misc/supplementary_material/supplementary_material.html).

To use this repository, please make sure to adapt the paths in `adversarial_ml/utils/paths.py`
accordingly, in particular the path pointing to the data (`d_path`), 
and to the `csv` files containing the labels (`csv_path`). 

Prerequisites
-------------
- [pytorch](https://pytorch.org/)
- [torchaudio](https://pytorch.org/audio/)
- [numpy](https://pypi.org/project/numpy/)
- [attrdict](https://pypi.org/project/attrdict/)
- [matplotlib](https://matplotlib.org/)
- [parse](https://pypi.org/project/parse/)

For details on the versions of the libraries we used, please 
view `adversarial_ml/requirements.txt`. The tested Python version is `3.7.3`.
If `conda` is available, a new environment
can be created and the necessary libraries installed via

````
conda create -n ad_env python==3.7.3 pip
conda activate ad_env
pip install -r requirements.txt
````

Data Setup
----------
The data we use is a part of the curated train set of the 
FSDKaggle2019 [1, 2] data. More precisely,
we use the 799 files that have one single of 12 different musical labels
(see `adversarial_ml/data/data_utils`) for more information on these labels.

To be able to run this code, download the [curated data](https://zenodo.org/record/3612637), 
and make sure to set the `d_path` in `adversarial_ml/paths.py` 
correctly, pointing to the extracted directory. 
After downloading the data, you might want to re-sample it to 16kHz
(e.g. with `torchaudio.transforms.Resample`), as we used 16kHz to train
the given pre-trained model.
Additionally, we need the labels which are available in 
`train_curated_post_competition.csv` 
(and `test_post_competition.csv`); for this, please set the 
`csv_path` to point to these files.


Project Structure
-----------------
The basic structure of this project should look like this:
```bash
adversarial_ml
├── adversarial_ml
│   ├── attacks
│   ├── baselines
│   ├── data
│   ├── evaluation
│   └── utils
├── misc
│   ├── pre_trained
│   └── supplementary_material
├── README.md
├── index.html
└── requirements.txt
```

In the sub-folder `aversarial_ml` the code can be found. The sub-directories are:
- `attacks`, contains the targeted methods C&W and Multi-Scale C&W, 
- `baselines`, contains the untargeted methods FGSM and PGDn,
- `data` contains any kind of data handling (e.g. datasets), 
- `evaluation` contains functions that help to evaluate the adversarial attacks,
- and `utils` contains our model architecture and all kind of utils.

The folder `misc/pre_trained` contains the trained CNN that we attack; 
in `misc/supplementary_material` the [supplementary material](https://prkat.github.io/adversarial_instrument_classification/misc/supplementary_material/supplementary_material.html) of above
described technical report is located (including hearing examples
and confusion matrices of the system after various attacks).

Adversarial examples will be stored in `misc/adversaries`, and
according logging-files in `misc/logs`.


Run Experiments
---------------

*FGSM*: To run this untargeted attack, define parameters in 
`adversarial_ml/baselines/fgsm_config.txt`. Then, you can run the 
python script, e.g. with

````
python3 -m adversarial_ml.baselines.fgsm
````

*PGDn*: The second untargeted attack can be run similarly. First,
define parameters in `adversarial_ml/baselines/pgdn_config.txt`, and
run the python script from command line with e.g.

````
python3 -m adversarial_ml.baselines.pgdn
````

*C&W* and *Multi-Scale C&W*: Both targeted attacks can be performed by running
````
python3 -m adversarial_ml.attacks.targeted_attack
````

To modify the parameter setting, change the according parameters in 
`adversarial_ml/attacks/attack_config.txt`. You can
switch between the two attacks by setting `attack = cw` or `attack = mscw`
respectively; in addition to that, the target class can be modified to
be either `target = random` or the name of a particular classs, e.g.
`target = accordion`.

Evaluation
----------
In order to evaluate your experiments, you can make use of 
functions provided in the `adversarial_ml/evaluation` directory:

- `confusion_matrices.py` allows you to plot confusion matrices;
- and `eval_funcs.py` contains methods to compute accuracies, the SNR,
confidences and iterations.


References 
----------

[1] Eduardo Fonseca, Manoj Plakal, Frederic Font, Daniel P. W. Ellis, 
 Xavier Serra. (2020). FSDKaggle2019 (Version 1.0) [Data set]. 
Zenodo. http://doi.org/10.5281/zenodo.3612637.

[2] Eduardo Fonseca, Manoj Plakal, Frederic Font, Daniel P. W. Ellis, 
Xavier Serra. "Audio tagging with noisy labels and minimal supervision". 
Proceedings of the DCASE 2019 Workshop, NYC, US (2019).

Acknowledgments
---------------
This work is supported by the Austrian National Science Foundation (FWF, P 31988).
