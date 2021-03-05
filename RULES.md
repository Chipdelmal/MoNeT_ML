# Objectives

* Interpretable ML results (sensitivity analysis)
* Scikit-learn compliant `joblib` file with trained model (one file per model)
* Provide GUIs to analyze the datasets and ML models whenever possible
* Classification should be flexible in resolution

# Rules

* Push to the repo often!
* Python code (as [PEP8](https://www.python.org/dev/peps/pep-0008/) compliant as possible)
* Document your code in [MD](https://www.markdownguide.org/) files and comments
* Git commits start with the 3 letter code of the project (PYF, STP, etc.)
* You can use [Jupyter](https://jupyter.org/) notebooks for exploration but the end result should be a `.py` set of files that can be run from the terminal
* These `py` files will be called from a `bash` command to create pipelines
* Functions definitions should have their own separate file(s)
* Have separate files for: data cleaning, training, testing, evaluation
* Have paths as clearly stated variables that can be changed easily
* Datasets will be synched through [Mega](https://mega.io/) for the time being
* Have a file with the re-scaling constants
* Return some type of uncertainty estimate