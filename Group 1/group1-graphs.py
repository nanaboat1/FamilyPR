###########################################
# program: create histogram of given data
# author: Jonathan Olderr
# date: 05/18/2021
# purpose: to get this final done
#############################################

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.reshape.pivot import crosstab

# configures plot settings
plt.rcParams.update({'figure.autolayout': True})

# import data
data = pd.read_csv("data_cleaned.csv")

# creating histogram crosstab
crosstab_1 = pd.crosstab( \
    data["X3.4.Race"], \
    data["X3.6.Gender"] \
)

crosstab_1.plot(kind = 'bar', stacked = True)

plt.title("Non-Normalized Gender vs Race frequency")
plt.xlabel("Race")
plt.ylabel("Count")
plt.savefig("non-normal gender and race frequency")

crosstab_1_norm = crosstab_1.div(crosstab_1.sum(axis = 1), axis = 0)
crosstab_1_norm.plot(kind = 'bar', stacked = True)

plt.title("Normalized Gender vs Race frequency")
plt.xlabel("Race")
plt.ylabel("Gender %Proportion%")
plt.savefig("normal gender and race frequency")

plt.clf()

crosstab_2 = pd.crosstab( \
    data["X3.5.Ethnicity"], \
    data["X3.6.Gender"] \
)

crosstab_2.plot(kind = 'bar', stacked = True)
plt.title("Non-Normalized Gender vs Ethnicity frequency")
plt.xlabel("Race")
plt.ylabel("Count")
plt.savefig("non-normal gender and Ethnicity frequency")

crosstab_2_norm = crosstab_2.div(crosstab_2.sum(axis = 1), axis = 0)
crosstab_2_norm.plot(kind = 'bar', stacked = True)

plt.title("Normalized Gender vs Race frequency")
plt.xlabel("Ethnicity")
plt.ylabel("Gender %Proportion%")
plt.savefig("normal gender and Ethnicity frequency")
