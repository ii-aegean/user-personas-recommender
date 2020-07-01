# User Personas Recommender System

This repository contains the code used in the experimental part of the paper [A Personalized Heritage-Oriented Recommender System Based on Extended Cultural Tourist Typologies](https://doi.org/10.3390/bdcc4020012) that appeared at the [Big Data Analytics for Cultural Heritage](https://www.mdpi.com/journal/BDCC/special_issues/data_heritage) Special Issue of the [Big Data and Cognitive Computing](https://www.mdpi.com/journal/BDCC) journal ([Volume 4, Issue 2](https://www.mdpi.com/2504-2289/4/2) - June 2020)


## Step 1 - Install requirements

The code is written in Python3 (version 3.6.7 or newer). Prior to running the provided scripts, install the modules that appear in the ```requirements.txt``` file (e.g. through ```pip install -r requirements.txt```).

## Step 2 - Download the data

Download the [Flickr User-POI Visits Dataset](https://sites.google.com/site/limkwanhui/datacode#h.p_ID_65) and unzip it to the ```data-ijcai15``` folder.

## Step 3 - Compute the cultural tourist profiles

Compute the cultural tourist profile of each visitor for the 8 cities of the dataset (```data-ijcai15/visits``` folder), using the ```profiles.py``` script. For example, for the city of Budapest, script parameters should be as follows 
```
python profiles.py -i data-ijcai15/visits/userVisits-Buda-allPOI.csv -o buda-profiles.csv
```

## Step 3 - Produce recommendations

Produce recommendations for both models discussed in the paper (baseline and user personas recommender), using the ```recommend.py``` script, along with the profiles generated at the previous step and the corresponding visitor data. 

For the same example city (Budapest) as before, script parameters should be as follows:
```
python recommend.py -d data-ijcai15/visits/userVisits-Buda-allPOI.csv -u buda-profiles.csv
```
and the output should be:
```
LightFM Model (Baseline)
------------------------
Precision: Train 47.27%, Test 30.33%
MRR: Train 76.10%, Test 55.53%

LightFM Model + User Personas
-----------------------------
Precision: Train 56.59%, Test 33.33%
MRR: Train 86.82%, Test 59.29%
```

The ```recommend.py``` script has a numer of optional parameters that affect the performance of the recommender system. In order to see them, along with their default values, run the script with the ```-h``` option
```
python recommend.py -h
```
The output should be as follows:
```
usage: recommend.py [-h] [-d D] [-u U] [--test TEST] [--seed SEED] [--f F]
                    [--lr LR] [--a A] [--loss LOSS] [--epochs EPOCHS] [--k K]

User Personas Recommender

optional arguments:
  -h, --help       show this help message and exit
  -d D             Flickr User-POI Visits file
  -u U             Profiles file
  --test TEST      Test percentage (default: 0.2)
  --seed SEED      Random seed (default: 2020)
  --f F            Number of features (default: 20)
  --lr LR          Learning rate (default: 0.05)
  --a A            L2 regularization parameter (user features) (default:
                   0.005)
  --loss LOSS      Loss function (default: bpr)
  --epochs EPOCHS  Number of epochs (default: 20)
  --k K            Recommendation list size (default: 3)
```

The default parameter values are according to the paper for the cities of Budapest, Delhi, Glasgow, Osaka and Perth. For the cities of Edinburgh, Toronto and Vienna, the L2 regularization parameter should be set to *0.002*, again according to the paper. 

For example, if ```edin-profiles.csv``` contains the cultural tourist profiles of the visitors of the city of Edinburgh, then the recommendation script should be run as follows
```
python recommend.py -d data-ijcai15/visits/userVisits-Edin.csv -u edin-profiles.csv --a 0.002
```
yielding the following output:
```
LightFM Model (Baseline)
------------------------
Precision: Train 51.57%, Test 37.02%
MRR: Train 82.92%, Test 67.27%

LightFM Model + User Personas
-----------------------------
Precision: Train 58.39%, Test 38.96%
MRR: Train 93.11%, Test 70.65%
```

The results appearing on Figures 5 & 6 of the paper are averages of 10 different runs, using 10 randomly selected initial random seeds (```--seed``` parameter of the ```recommend.py``` script).

## License & Citations

The source code is provided under an Apache License, Version 2 (please read ```LICENSE.txt``` for more details). If you plan to use this code in your project or research, please cite the following publication:
```
@article{Konstantakis_2020, 
title={A Personalized Heritage-Oriented Recommender System Based on Extended Cultural Tourist Typologies}, 
 volume={4}, 
 ISSN={2504-2289}, 
 url={http://dx.doi.org/10.3390/bdcc4020012}, 
 DOI={10.3390/bdcc4020012}, 
 number={2}, 
 journal={Big Data and Cognitive Computing}, 
 publisher={MDPI AG}, 
 author={Konstantakis, Markos and Alexandridis, Georgios and Caridakis, George}, 
 year={2020}, 
 month={Jun}, 
 pages={12}
}
```