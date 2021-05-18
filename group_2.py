##############
# names: James Holtz and Alaina Stockdill
# project: Family promise project
# date: 
##############

##############
# sources:
#
# Example for how to select rows with specific condition
# https://chrisalbon.com/python/data_wrangling/pandas_selecting_rows_on_conditions/
#
##############

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
#import statsmodels.tools.tools as stattools
import random

# Data set from Family Promise 2021
data = pd.read_csv("./FAM_PROM_REDACTED_2021.csv")

# Rename all of the columns
# Code from J Wylie over google collaborate
data = data.rename(columns={
                        '4.3 Non-Cash Benefit Count':'non-cash_benefit_count',
                        '4.3 Non-Cash Benefit Count at Exit':'non-cash_count_at_exit',
                        'CurrentDate':'current_date','2.1 Organization Name':'org_name',
                        '2.2 Project Name':'project_name',
                        '2.4 ProjectType':'project_type',
                        '2.5 Utilization Tracking Method (Invalid)':'util_track_method',
                        '2.6 Federal Grant Programs':'fed_grant_programs',
                        'Enrollment Created By':'enrollment_created_by',
                        '3.1 FirstName':'first_name','3.1 LastName':'last_name',
                        '5.8 Personal ID':'personal_id','5.9 Household ID':'case_id',
                        '3.15 Relationship to HoH':'relationship_to_HoH',
                        '3.16 Client Location':'client_location','CaseMembers':'case_members',
                        '3.10 Enroll Date':'enroll_date','3.11 Exit Date':'exit_date',
                        '3.12 Exit Destination':'exit_destination','3.2 SocSecNo':'ssn',
                        '3.2 Social Security Quality':'ssn_quality','3.3 Birthdate':'dob',
                        '3.3 Birthdate Quality':'dob_quality','Age at Enrollment':'age_at_enrollment',
                        'Current Age':'current_age','3.4 Race':'race','3.5 Ethnicity':'ethnicity',
                        '3.6 Gender':'gender','3.7 Veteran Status':'vet_status',
                        '3.8 Disabling Condition at Entry':'disabling_cond_at_entry',
                        '3.917 Living Situation':'living_situation',
                        '3.917b Stayed Less Than 7 Nights':'stayed_7_or_less',
                        '3.917 Stayed Less Than 90 Days':'stayed_90_or_less',
                        '3.917b Stayed in Streets, ES or SH Night Before':'stayed_street_ES_or_SH_night_before',
                        '3.917 Length of Stay':'length_of_stay',
                        '3.917 Homeless Start Date':'homeless_start_date',
                        'Length of Time Homeless (3.917 Approximate Start)':'length_of_time_homeless',
                        '3.917 Times Homeless Last 3 Years':'times_homeless_last_3years',
                        '3.917 Total Months Homeless Last 3 Years':'total_months_homeless',
                        'V5 Last Permanent Address':'last_perm_address',
                        'V5 Prior Address':'prior_address','V5 State':'state',
                        'V5 Zip':'zip','Municipality (City or County)':'municipality',
                        'Days Enrolled in Project':'days_enrolled_in_project',
                        'RRH In Permanent Housing':'rrh_in_perm_housing',
                        'RRH Date Of Move-In':'rrh_date_of_move-in',
                        'Days Enrolled Until RRH Date of Move-in':'days_enrolled_until_rrh_movein',
                        '4.1 Housing Status':'housing_status',
                        '4.4 Covered by Health Insurance':'covered_by_health_insurance',
                        '4.11 Domestic Violence':'domestic_violence',
                        '4.11 Domestic Violence - Currently Fleeing DV?':'currently_fleeing',
                        '4.11 Domestic Violence - When it Occurred':'when_dv_occured',
                        '4.13 Engagement Date':'engagement_date',
                        'Days Enrolled Until Engagement Date':'days_enrolled_until_engaged',
                        '4.24 Current Status (Retired Data Element)':'current_status',
                        '4.24 In School (Retired Data Element)':'in_school',
                        '4.24 Connected to McKinney Vento Liason (Retired)':'connected_to_MVento',
                        'Household Type':'household_type',
                        'Latitude':'latitude','Longitude':'longitude',
                        'R1 Referral Source':'referal_source',
                        'R2 Date Status Determined':'date_status_determined',
                        'R2 Enroll Status':'enroll_status',
                        'R2 Runaway Youth':'runaway_youth',
                        'R2 Reason Why No Services Funded':'reason_why_no_services_funded',
                        'R3 Sexual Orientation':'sexual_orientation',
                        'R4 Last Grade Completed':'last_grade_completed',
                        'R5 School Status':'school_status',
                        'R6 Employed Status':'employed_status',
                        'R6 Why Not Employed':'reason_not_employed',
                        'R6 Type of Employment':'type_of_employment',
                        'R6 Looking for Work':'looking_for_work',
                        'R7 General Health Status':'general_health_status',
                        'R8 Dental Health Status':'dental_health_status',
                        'R9 Mental Health Status':'mental_health_status',
                        'R10 Pregnancy Status':'pregnancy_status',
                        'R10 Pregnancy Due Date':'pregnancy_due_date',
                        'Client Record Restricted':'client_record_restricted',
                        'InfoReleaseNo':'infoReleaseNo',
                        'Information Release Status':'info_release_status',
                        '4.12 Contact Services':'contact_services',
                        'Date of Last Contact (Beta)':'date_of_last_contact',
                        'Date of First Contact (Beta)':'date_of_first_contact',
                        'Count of Bed Nights (Housing Check-ins)':'housing_checkins',
                        'Date of Last ES Stay (Beta)':'date_of_last_stay',
                        'Date of First ES Stay (Beta)':'date_of_first_stay',
                        '4.2 Income Total at Entry':'income_at_entry',
                        '4.02 Total Income at Annual Update':'income_at_update',
                        '4.2 Income Total at Exit':'income_at_exit',
                        'Barrier Count at Entry':'barrier_count_at_entry',
                        'Chronic Homeless Status':'chronic_homeless_status',
                        'ProgramType':'program_type',
                        'SOAR Eligibility Determination (Most Recent)':'soar_eligibility',
                        'SOAR Enrollment Determination (Most Recent)':'soar_enrollment',
                        'RRH | Most Recent Enrollment':'most_recent_rrh',
                        'Street Outreach | Most Recent Enrollment':'most_recent_street_outreach',
                        'Coordinated Entry | Most Recent Enrollment':'most_recent_CE',
                        'Emergency Shelter | Most Recent Enrollment':'most_recent_ES',
                        'Transitional Housing | Most Recent Enrollment':'most_recent_trans',
                        'PSH | Most Recent Enrollment':'most_recent_PSH',
                        'Prevention | Most Recent Enrollment':'most_recent_prevention',
                        'Under 25 Years Old':'under_25',
                        'ClientID':'client_id',
                        '4.10 Alcohol Abuse (Substance Abuse)':'alcohol_abuse',
                        '4.07 Chronic Health Condition':'chronic_health_condition',
                        '4.06 Developmental Disability':'developmental_disability',
                        '4.10 Drug Abuse (Substance Abuse)':'substance_abuse',
                        '4.08 HIV/AIDS':'HIV_AIDS',
                        '4.09 Mental Health Problem':'mental_health_problem',
                        '4.05 Physical Disability':'physical_disability',
                        'CaseChildren':'case_children','CaseAdults':'case_adults',
                        'Bed Nights During Report Period':'bednights_during_report_period',
                        'Count of Bed Nights - Entire Episode':'entire_episode_bednights',
                        'HEN-HP Referral Most Recent':'most_recent_HEN-HP',
                        'HEN-RRH Referral Most Recent':'most_recent_HEN-RRH',
                        'WorkSource Referral Most Recent':'most_recent_worksource',
                        'YAHP Referral Most Recent':'most_recent_YAHP',
                        '4.04.10 Other Public':'other_public',
                        '4.04.10 State Funded':'state_funded',
                        '4.04.11 Indian Health Services (IHS)':'indian_health_services',
                        '4.04.12 Other':'other',
                        '4.04.3 Combined Childrens HealthInsurance/Medicaid':'combined_childrens_health_insurance',
                        '4.04.3 Medicaid':'medicaid','4.04.4 Medicare':'medicare',
                        "4.04.5 State Children's health Insurance S-CHIP":'CHIP',
                        "4.04.6 Veteran's Administration Medical Services":'VAMS',
                        '4.04.8 Health Insurance obtained through COBRA':'COBRA',
                        '4.04.7 Private - Employer':'Private_employer',
                        '4.04.9 Private':'private','4.04.9 Private - Individual':'private_individual',
                        '4.2.3a Earned Income':'earned_income',
                        '4.2.4b Unemployment Insurance':'unemployement_income',
                        '4.2.5c Supplemental Security Income':'supplemental_security_income',
                        '4.2.6d Social Security Disability  Income':'social_security_disability_income',
                        '4.2.7e VA Disability Compensation':'VA_disability_compensation',
                        '4.2.8f VA Disability Pension':'VA_disability_pension',
                        '4.2.9g Private Disability Income':'private_disability_income',
                        '4.2.10h Workers Compensation':'workers_compensation',
                        '4.2.11i TANF':'TANF','4.2.12j General Assistance':'general_assistance',
                        '4.2.13k Retirement (Social Security)':'retirement_social_security',
                        '4.2.14l Pension from a Former Job':'pension_from_former_job',
                        '4.2.15m Child Support':'child_support',
                        '4.2.16n Alimony':'alimony',
                        '4.2.17o Other Income':'other_income',
                        'Chronic Homeless Status_vHMISDatAssessment':'chronic_homeless_status_assessment',
                        'Chronic Homeless Status_EvaluatevHMIS&HMISDA':'chronic_homeless_status_evaluation',
                        'Email':'email','HomePhone':'home_phone','WorkPhone':'work_phone'})

# Convert column names to lower case
data.columns = map(str.lower, data.columns)

#------------------------Create our subset-----------------------
# Create a subset of data with just the clients that are self or significant other
# First with both spouse and significant other added
"""
self_subset = data[(data['relationship_to_hoh'] == "Self") | (data['relationship_to_hoh'] == "Significant Other (Non-Married)")\
    | (data['relationship_to_hoh'] == "Spouse")]

self_subset = self_subset[(self_subset['case_members'] != np.nan) & (self_subset['length_of_time_homeless'] != np.nan) &\
    (self_subset['days_enrolled_in_project'] != np.nan) &\
    (self_subset['income_at_exit'] != np.nan) & (self_subset['entire_episode_bednights'] != np.nan)]
"""

# Subset with relationship_to_hoh of "Self" only
self_subset = data[(data['relationship_to_hoh'] == "Self")]

# Replace the empty exit destinations with "Not exited"
self_subset['exit_destination'].fillna("Not exited", inplace = True)

# Replace empty incomes with 0 - both income fields
# Soucre: https://pythoninoffice.com/create-calculated-columns-in-a-dataframe/
self_subset['income_at_entry'].fillna("0", inplace = True)
self_subset['income_at_exit'].fillna("0", inplace = True)

# The 0's in length of stay should be filled with the average
# FIND SOURCE FOR THIS
self_subset['length_of_time_homeless'].fillna(self_subset['length_of_time_homeless'].mean(), inplace = True)

# Make the values in income fields floats to be able to calculate change in income
# Source: https://www.kite.com/python/answers/how-to-convert-a-pandas-dataframe-column-of-strings-to-floats-in-python
self_subset['income_at_entry'] = pd.to_numeric(self_subset['income_at_entry'], downcast = "float")
self_subset['income_at_exit'] = pd.to_numeric(self_subset['income_at_exit'], downcast = "float")

# Create column of the change in income between entry and exit
self_subset['change_in_income'] = self_subset['income_at_exit'] - self_subset['income_at_entry']


# Create a new column called housing that will store simplified exit destination names
self_subset['housing'] = self_subset['exit_destination']
dict_housing = {"housing":{
                    "Rental by client, other ongoing housing subsidy": "Rental",
                    "Rental by client with RRH or equivalent subsidy": "Rental",
                    "Place not meant for habitation (e.g., a vehicle, an abandoned building, bus/train/subway station/airport or anywhere outside)": "Not meant for habitation",
                    "Staying or living with family, temporary tenure (e.g., room, apartment or house)":"Temporary family/friends",
                    "Staying or living with family, permanent tenure": "Permanent family/friends",
                    "Staying or living with friends, permanent tenure":"Permanent family/friends",
                    "Staying or living with friends, temporary tenure (e.g., room, apartment or house)":"Temporary family/friends",
                    "Transitional Housing for homeless persons (including homeless youth)":"Transitional housing",
                    "Emergency shelter, including hotel or motel paid for with emergency shelter voucher, or RHY-funded Host Home shelter":"Emergency housing",
                    "Hotel or Motel paid for without Emergency Shelter Voucher":"Hotel",
                    "Client doesn\'t know":"Client doesnt know"
                    }
                }
self_subset.replace(dict_housing, inplace = True)

# Alphabetize the dataframe using housing column - this will help later on
# Source: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
self_subset.sort_values(by = ['housing'], ascending = True, inplace = True)

# Create a training and test data split
self_subset_train, self_subset_test = train_test_split(self_subset, test_size = 0.25, random_state = 23)

##########################################################
# Decision Trees
##########################################################

# ------------------------ CART Analysis --------------------------
# Target variable is categorical so we need dummy variables
#yaxis = self_subset_train[['housing']]
Y = self_subset_train[['housing']]

# Alphabetize the dataframe using housing column - this will help later on
# Shouldn't need this because we can use the feature and class names built in
#Y.sort_values(by=['housing'], ascending = True, inplace = True)

# Get all of the predictor variables
X = pd.concat([ self_subset_train['case_members'], self_subset_train['length_of_time_homeless'],\
    self_subset_train['days_enrolled_in_project'], self_subset_train['income_at_exit'], \
    self_subset_train['entire_episode_bednights'], self_subset_train['change_in_income']], axis = 1)

print(Y.shape)
print(X.shape)

# Create the CART tree classifier
cart01 = DecisionTreeClassifier(criterion = "gini", max_leaf_nodes = 7).fit(X,Y)
# Create the decision tree for the cart anlaysis
# How to get feature and class names from source below
# Soucre: https://stackoverflow.com/questions/39476020/get-feature-and-class-names-into-decision-tree-using-export-graphviz
dot_data = tree.export_graphviz(cart01, out_file = "cart_01.dot", filled = True, rounded = True,
                special_characters = True, feature_names = X.columns[0:], class_names = cart01.classes_ )

# Run the cart model on the testing data
# Run the cart model on the testing data
# First need to create X_test, and Y_test
Y_test = self_subset_test[['housing']]
X_test = pd.concat([ self_subset_test['case_members'], self_subset_test['length_of_time_homeless'], \
                     self_subset_test['days_enrolled_in_project'], self_subset_test['income_at_exit'], \
                     self_subset_test['entire_episode_bednights'], self_subset_test['change_in_income']], axis = 1)

# Predict the housing of the test set
pred_housing01 = cart01.predict(X_test)   

# Create a contingency table to access the accuracy of the model
# Compares the actual regional indicator to the regional indicator predicted using the model
crosstab_01 = pd.crosstab(Y_test['housing'], pred_housing01)

# Calculate the percentages for each column
crosstab_01b = round(crosstab_01.div(crosstab_01.sum(0), axis = 1)*100, 1)

# Prints out the two contingency tables to check accuracy
print(crosstab_01)
print (crosstab_01b)

# calculate accuracy and print
YT = Y_test.values
count = 0.
correct = 0.
for i in range(0, len(YT)):
    if YT[i]==pred_housing01[i]:
        correct += 1
    count +=1
print("\nCart01 accuracy:")
print(correct/count*100)
print("\n\n")

# --------------------------- Just exits -------------------------
# Create a new decision tree with just those that have exited
exits = self_subset[(self_subset['housing'] != "Not exited")]

# Create a training and testing set of just the exits
exits_train, exits_test = train_test_split(exits, test_size = 0.25, random_state = 23)

# Re-run CART analysis with new subset
# Target varaible of exit destination
Y_exits = exits_train[['housing']]

# Get the predictor variables
X_exits = pd.concat([ exits_train['case_members'], exits_train['length_of_time_homeless'],\
    exits_train['days_enrolled_in_project'], exits_train['income_at_exit'], \
    exits_train['entire_episode_bednights'], exits_train['change_in_income']], axis = 1)

# Create the CART tree classifier for the exit data
cart02 = DecisionTreeClassifier(criterion = "gini", max_leaf_nodes = 7).fit(X_exits,Y_exits)
dot_data = tree.export_graphviz(cart02, out_file = "cart_02.dot", filled = True, rounded = True, \
         special_characters = True, feature_names = X_exits.columns[0:], class_names = cart02.classes_ )

# Get the X and Y sets to run the prediction 
Y_exits_test = exits_test[['housing']]
X_exits_test = pd.concat([ exits_test['case_members'], exits_test['length_of_time_homeless'], \
                     exits_test['days_enrolled_in_project'], exits_test['income_at_exit'], \
                     exits_test['entire_episode_bednights'], exits_test['change_in_income']], axis = 1)

# Run the cart model on the testing data for just the exits
pred_housing02 = cart02.predict(X_exits_test)    

# Create a contingency table to access the accuracy of the model
# Compares the actual regional indicator to the regional indicator predicted using the model
crosstab_02 = pd.crosstab(Y_exits_test['housing'], pred_housing02)

# Calculate the percentages for each column
crosstab_02b = round(crosstab_02.div(crosstab_02.sum(0), axis = 1)*100, 1)

# Prints out the two contingency tables
print(crosstab_02)
print (crosstab_02b)

# calculate accuracy and print
YT = Y_exits_test.values
count = 0.
correct = 0.
for i in range(0, len(YT)):
    if YT[i]==pred_housing02[i]:
        correct += 1
    count +=1
print("\nCart02 accuracy:")
print(correct/count*100)
print("\n\n")


# ------------------------ C5.0 Analysis --------------------------
c50_01 = DecisionTreeClassifier(criterion = "entropy", max_leaf_nodes = 7).fit(X,Y)

c50_plot_01 = tree.export_graphviz(c50_01, out_file = "c5_01.dot", filled = True, rounded = True,
                special_characters = True, feature_names = X.columns[0:], class_names = c50_01.classes_ )
        
# Check the accuracy of the C5.0 model for all self data
pred_housing03 = c50_01.predict(X_test)    

# Create a contingency table to access the accuracy of the model
# Compares the actual regional indicator to the regional indicator predicted using the model
crosstab_03 = pd.crosstab(Y_test['housing'], pred_housing03)

# Calculate the percentages for each column
crosstab_03b = round(crosstab_03.div(crosstab_03.sum(0), axis = 1)*100, 1)

# Prints out the two contingency tables
print(crosstab_03)
print (crosstab_03b)

# calculate accuracy and print
YT = Y_test.values
count = 0.
correct = 0.
for i in range(0, len(YT)):
    if YT[i]==pred_housing03[i]:
        correct += 1
    count +=1
print("\nC5_01 accuracy:")
print(correct/count*100)
print("\n\n")

# Create a c5.0 decision tree for the just exits data
c50_02 = DecisionTreeClassifier(criterion = "entropy", max_leaf_nodes = 7).fit(X_exits,Y_exits)
c50_plot_02 = tree.export_graphviz(c50_02, out_file = "c5_02.dot", filled = True, rounded = True,
                special_characters = True, feature_names = X_exits.columns[0:], class_names = c50_02.classes_  )

# Check the accuracy of the C5.0 model for all self data
pred_housing04 = c50_02.predict(X_exits_test)    

# Create a contingency table to access the accuracy of the model
# Compares the actual regional indicator to the regional indicator predicted using the model
crosstab_04 = pd.crosstab(Y_exits_test['housing'], pred_housing04)

# Calculate the percentages for each column
crosstab_04b = round(crosstab_04.div(crosstab_04.sum(0), axis = 1)*100, 1)

# Prints out the two contingency tables
print(crosstab_04)
print (crosstab_04b)

# calculate accuracy and print
YT = Y_exits_test.values
count = 0.
correct = 0.
for i in range(0, len(YT)):
    if YT[i]==pred_housing04[i]:
        correct += 1
    count +=1
print("\nC5_02 accuracy:")
print(correct/count*100)
print("\n\n")

# ------------------------- Neural Net --------------------------
# Create the model
nn_model = Sequential() 
nn_model.add(Dense(4, input_dim = 6, activation = 'sigmoid'))
nn_model.add(Dense(4, activation = 'sigmoid'))
nn_model.add(Dense(8, activation = 'sigmoid'))

# Compile the keras model
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# use get_dummies to break the Y into classes
Y_exits_nn = pd.get_dummies(Y_exits)

# Fit the keras model on the dataset
nn_model.fit(X_exits, Y_exits_nn, epochs = 150, batch_size = 10)

# set-up the test data and evaluate the model
Y_exits_test_nn = pd.get_dummies(Y_exits_test)
nn_model.evaluate(X_exits_test,Y_exits_test_nn)

