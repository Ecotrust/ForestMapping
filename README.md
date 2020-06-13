Forest Mapping
==============

Predicting Pacific Northwest forest types from remotely-sensed data.

This repository includes data cleaning, model-fitting, and applications of
predictive models to estimate basic forest attributes using lidar data,
satellite and aerial imagery, and down-scaled climate information.

--------------------

This effort has been supported by two Conservation Innovation Grants from the
USDA Oregon Natural Resources Conservation Service:  

- "Technology Transfer for Rapid Family Forest Assessment and Stewardship
  Planning" - FY 2017 Oregon Conservation Innovation Grant, Award
  # 69-0436-17-036.  
- "Modern Land Mapping Toolkit to Streamline Forest Stewardship Planning" -  
  FY 2019 Oregon Conservation Innovation Grant, Award # NR190436XXXXG012

This effort has also been supported by a grant of cloud storage and computing
services made available to Ecotrust under the Microsoft <a target="_blank"
href="https://www.microsoft.com/en-us/ai/ai-for-earth/">AI for Earth Program</a>
in a project entitled "Mining Public Datasets to Automate Forest Stand
Delineation and Labeling."


Project Organization
--------------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Working versions of data during processing.
    │   ├── processed      <- Processed datasets ready for modeling.
    │   └── raw            <- Raw data
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Reports (PDF, etc.)
    │   └── figures        <- Generated graphics and figures used in reports.
    │
    ├── environment.yml    <- The requirements file for reproducing the analysis environment, e.g.
    │                         using `conda create env --file environment.yml`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │                     predictions
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualization


--------

<p><small>Project organization based on the <a target="_blank"
href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter
data science project template</a>. #cookiecutterdatascience</small></p>
