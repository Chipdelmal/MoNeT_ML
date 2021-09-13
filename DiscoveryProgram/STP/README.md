# Discovery Program: Mosquito Gene Drive Machine Learning Library

Our group has been generating datasets on the expected results on deploying genetically-modified mosquitoes over the last year. One of the challenges, however, is to share our results with collaborators and stakeholders. Installing and running our Mosquito Gene Drive Explorer (MGDrivE) model takes time and requires lots of computational resources; as such, we are working on generating surrogate machine learning models that are trained upon a set of summary statistics that describe the behavior of the system.

## Goals

1. To provide a report on the data exploration for the gene drive datasets provided.
   *  Correlation analysis
   *  Data distributions
   *  PCA
   *  Encoding
   *  Speedup DICE plots*
   *  Comparison between LDR/SDR
2. To train various version of regression and classification models for the datasets.
   * Check for overfits
   * Training/Validation
   * Auto-generate ML reports for the trained models (R-squared, confusion matrix, mean squared error, etc)
3. To create and upload a GUI with the models and predictions, so that other teams can explore the responses.
   * Check previous drafts by [Chris De Leon](https://mgdrive.herokuapp.com/) and [Elijah Bartolome](https://share.streamlit.io/elijahbartolome/monet_ml/main/STP/RBC/web_ui.py)
   * Explore alternatives

## Datasets Description

These datasets were generated as part of a larger publication on the effects of simulating the releases of CRISPR/Cas9 linked and split drive genetic modifications in the islands of São Tomé and Príncipe in equatorial Africa. In these versions of the dataset, we consider the landscape as a fully-mixing panmictic population to focus on the inherent properties of the drives, without the spatial component of the migration of the mosquitos.


<img src="../../media/centrality.jpg" style="width:30%;"><br>

### Inputs (Features)

* **i_sex** [1, 2, 3]: Sex-sorting of the released transgenic mosquitos [1]
* **i_ren** [0 _to_ 24]: Number of releases (weekly) [12]
* **i_res** [0 _to_ 1]: Release size (fraction of the total population) [0.5]
* **i_rsg** [0 _to_ 0.1185]: Resistance generation rate [0.079]
* **i_gsv** [0 _to_ 1e-2]: Genetic standing variation [1e-2]
* **i_fch** [0 _to_ 1]: Fitness cost on the H alleles (homing) [0.175]
* **i_fcb** [0 _to_ 1]: Fitness cost on the B alleles (out-of-frame resistant) [0.117]
* **i_fcr** [0 _to_ 1]: Fitness cost on the R alleles (in-frame resistant) [0]
* **i_hrm** [0 _to_ 1]: Homing rate on males [1]
* **i_hrf** [0 _to_ 1]: Homing rate on females [0.956]
* **i_grp** [0]: Group (unused)
* **i_mig** [0]: Migration rate (unused)



### Outputs (Labels)

* **TTI**: Time to Introgression
* **TTO**: Time to Outrogression*
* **WOP**: Window of Protection
* **POE**: Probability of Elimination
* **POF**: Probability of Fixation*
* **CPT**: Cumulative Potential for Transmission
* **MNF**: Minimum Unmodified Population*

The summary statistics marked with '*' are not relevant for these datasets, or need fixing.

<img src="../../media/stats.jpg" style="width:75%;"><br>

## Rules

* Document your work constantly!
* Document the reasons behind the decisions made.
* Push your work often to the repo!
* Always think about what others will think when they read your code.
* Think about interpretability whenever making decisions on the models.
* Work on the designated folder of the repo (the one in which this README file is located at).
* Do not push datasets or models to the repo! (only code).
* Export your models as [joblib](https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html) objects.
* You can use [Jupyter](https://jupyter.org/) code for development and exploration, but the end product must be a set pt **.py** files that can be called in series from the terminal (they will be run on a server).
* Functions definitions should have their own separate file(s).
* Have separate files for: data cleaning, training, testing, evaluation (pipelines).
* Have input/output paths as clearly stated variables that can be changed easily.
* Auto-generate ML reports for the trained models (R-squared, confusion matrix, etc).


<img src="../../media/MoNeT.jpg" style="width:100%;"><br>

## Team Roster

* Active: Lillian Weng, Joanna Yoo, Xingli Yu, Ayden Salazar
* Related Projects: [Elijah Bartolome](https://share.streamlit.io/elijahbartolome/monet_ml/main/STP/RBC/web_ui.py), [Christopher De Leon](https://mgdrive.herokuapp.com/)
* Lead: [Héctor M. Sánchez C.](https://chipdelmal.github.io/blog/)