# Discovery Program: Mosquito Gene Drive Machine Learning Library

Our group has been generating datasets on the expected results on deploying genetically-modified mosquitoes over the last year. One of the challenges, however, is to share our results with collaborators and stakeholders. Installing and running our Mosquito Gene Drive Explorer (MGDrivE) model takes time and requires lots of computational resources; as such, we are working on generating surrogate machine learning models that are trained upon a set of summary statistics that describe the behavior of the system.

## Goals

1. To provide a report on the data exploration for the gene drive datasets provided.
   1. Correlation analysis
   2. Data distributions
2. To train various version of regression and classification models for the datasets.
3. To create and upload a GUI with the models and predictions, so that other teams can explore the responses.

## Datasets Description

These datasets were generated as part of a larger publication on the effects of simulating the releases of CRISPR/Cas9 linked and split drive genetic modifications in the islands of São Tomé and Príncipe in equatorial Africa. In these versions of the dataset, we consider the landscape as a fully-mixing panmictic population to focus on the inherent properties of the drives, without the spatial component of the migration of the mosquitos.


<img src="../../media/centrality.jpg" style="width:25%;"><br>

### Inputs (Features)

* **i_sex**: Sex-sorting of the released transgenic mosquitos
* **i_ren**: Number of releases (weekly)
* **i_res**: Release size (fraction of the total population)
* **i_rsg**: Resistance generation rate
* **i_gsv**: Genetic standing variation
* **i_fch**: Fitness cost on the H alleles (homing)
* **i_fcb**: Fitness cost on the B alleles (out-of-frame resistant)
* **i_fcr**: Fitness cost on the R alleles (in-frame resistant)
* **i_hrm**: Homing rate on males
* **i_hrf**: Homing rate on females
* **i_grp**: Group (unused)
* **i_mig**: Migration rate (unused)

### Outputs (Labels)

* **TTI**: Time to Introgression
* **TTO**: Time to Outrogression
* **WOP**: Window of Protection
* **POE**: Probability of Elimination
* **POF**: Probability of Fixation
* **CPT**: Cumulative Potential for Transmission
* **MNF**: Minimum Unmodified Population

<img src="../../media/stats.jpg" style="width:75%;"><br>

## Rules

* Push your work often to the repo!
* Work on the designated folder of the repo (the one in which this README file is located at).
* Do not push datasets or models to the repo! (only code).
* Always think about what others will think when they read your code.
* Export your models as [joblib](https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html) objects.
* You can use [Jupyter](https://jupyter.org/) code for development and exploration, but the end product must be a set pt **.py** files that can be called in series from the terminal (they will be run on a server).
* Functions definitions should have their own separate file(s).
* Have separate files for: data cleaning, training, testing, evaluation (pipelines).
* Have input/output paths as clearly stated variables that can be changed easily.
* Auto-generate ML reports for the trained models (R-squared, confusion matrix, etc).
* Think about interpretability whenever making decisions on the models.

## Team Roster

* Lillian Weng
* Joanna Yoo
* Xingli Yu
* Ayden Salazar
* Héctor M. Sánchez C.
## Former Team

* Elijah Bartolome, Christopher De Leon
