# Spatial clustering in vaccination hesitancy: the role of social influence and social selection

This repository holds the processed data, modeling code, and figures for the following research: 
“Spatial clustering in vaccination hesitancy: the role of social influence and social selection”
Lucila G Alvarez-Zuzek, Casey Zipfel, Shweta Bansal
The preprint is available at doi: https://doi.org/10.1101/2022.01.11.22269032

## How to use this resource

The repository contains the simulators (Python scripts) that generate the different social processes scenarios: social selection, social influence, and the combined scenario.

email: sb753@georgetown.edu, la766@georgetown.edu

Please cite the paper above, if you use our code in any form or create a derivative work.


## Code (scripts/)

Simulators within each folder: social_influence, social_selection and combined_processes, are to generate the stochastic simulations for the corresponding social process. Inputs are set in the code (network and process characteristics, for instance spatiality). Scripts to process and aggregate the four datasets: spatial proximity and social connectedness networks, hesitancy and traits county's attributes, and stochastic simulator to validate results are found in /data folder. 

## Outputs

Within each folder there is a jupyter notebook file (.ipynb) that process simulator results and generates the corresponding figures.

## Sample Output

![alt tag](https://github.com/luzuzek/spatialclustering_homophily_influence/blob/03fd1b9fe82d6982bb66e68d71e153b943b4b466/data/Fig4%202.png)
