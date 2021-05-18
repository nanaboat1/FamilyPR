##############################
# Authors: Jonathan Olderr
# Date: 05/18/2021
# Program: 
##############################

#install packages
install.packages("jtools")
install.packages("ggplot2")
install.packages("gridExtra")
install.packages("e1071")
install.packages("C50")
install.packages("plyr")
install.packages("proxyC")

#load library's
library(jtools)
library(ggplot2)
library(gridExtra)
library(e1071)
library(C50)
library(plyr)
library(proxyC)

###############################################
# Create a function to facilitate crosstable evaluation
# * cross_tab --> the crosstable we are wanting to evaluate
# * beta --> weighting of the recall and precision
# * model_name --> Model name
# Returns a dataframe with important measurements
###############################################
model_eval <- function( cross_tab, beta, model_name, binary) {
  
  test <- 0
  TP <- 0
  FP <- 0
  FN <- 0
  TN <- 0
  
  if(binary) {
    TN <- cross_tab[1, 1]
    FN <- cross_tab[2, 1]
    TP <- cross_tab[2, 2]
    FP <- cross_tab[1, 2]
  }
  else { # yeah I had to hard code it in, real bummer
    for(x in 1:5) {
      TP <- TP + cross_tab[x, x]
      FP <- FP + sum(cross_tab[, x]) - TP
      FN <- FN + sum(cross_tab[x, ]) - TP
    }
    TN <- sum(rowSums(cross_tab == 0))
  }
  
  
  TAN <- TN + FP              # Total actually Negative
  TAP <- FN + TP              # Total actually Positive
  TPN <- TN + FN              # Total Predicted Negative
  TPP <- TP + FP              # Total Predicted Positive
  GT <- TP + FP + FN + TN     # Grand Total
  
  Accuracy <-  (TN + TP) / GT # Percentage predicted correctly overall
  ErrorRate <- (FN + FP) / GT # Error Rate = 1 - Accuracy
  Sensitivity <- (TP/TAP)     # ability to classify positively
  Recall <- (TN/TAN)          # Percentage negative prediction accuracy 
  Specificity <- Recall       # ability to classify negatively                 
  Precision <- (TP/TPP)       # percentage  of positive predictions that were accurate
  F_Beta <- ((1+beta*beta)*Precision*Recall) / ((beta*beta*Precision) + Recall )
  F_1 <- ((2)*Precision*Recall) / ((Precision) + Recall)
  F_2 <- ((5)*Precision*Recall) / ((4*Precision) + Recall)
  F_p5 <- ((1.25)*Precision*Recall) / ((0.25*Precision) + Recall)
  
  # creating array for storage of metrics
  metrics <- c(round(Accuracy,4), round(ErrorRate,4), round(Sensitivity,4), round(Specificity,4),
                round(Precision,4), round(beta,4), round(F_Beta,4), round(F_1, 4), round(F_2, 4),
                round(F_p5, 4))
  
  # sorting evaluation data
  eval_data <- data.frame(metrics)
  
  #eval_data <- as.data.frame(eval_data)
  
  colnames(eval_data) <- c(model_name)
  rownames(eval_data) <- c("Accuracy", "Error Rate", "Sensitivity", "Specificity/Recall", "Precision", "Beta", "F_Beta", "F1", "F2", "F0.5")  
  #return(TN)
  return(eval_data)
}

# set working directory
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

# input data
data <- read.csv("data_cleaned.csv")

# clean up data
data_clean <- data.frame(data$Age.at.Enrollment, data$X3.4.Race, data$X3.5.Ethnicity, data$X3.6.Gender, data$Days.Enrolled.in.Project)
colnames(data_clean) <- c("Age", "Race", "Ethnicity", "Gender", "Days_Enrolled")
data_clean[data_clean == "Gender Non-Conforming (i.e. not exclusively male or female)"] <- NA
# creating dummy variables
data_clean$Race_American_Indian_or_Alaska_Native <- ifelse(data_clean$Race == "American Indian or Alaska Native", 1, 0)
data_clean$Race_White <- ifelse(data_clean$Race == "White", 1, 0)
data_clean$Race_Multi_Racial <- ifelse(data_clean$Race == "Multi-Racial", 1, 0)
data_clean$Race_Black_or_African_American <- ifelse(data_clean$Race == "Black or African American", 1, 0)
data_clean$Race_Native_Hawaiian_or_Other_Pacific_Islander <- ifelse(data_clean$Race == "Native Hawaiian or Other Pacific Islander", 1, 0)
# clear NA values
data_clean <- na.omit(data_clean)

# examine age vs. Days in program
linear_model_Days = lm( formula = Age ~ Days_Enrolled, data_clean)
# create plot
plot_Days <- ggplot(data_clean, aes(x = Days_Enrolled, y = Age)) + 
  geom_point(size = 2) +
  ggtitle(label = "Days Enrolled vs. Age") +
  xlab("Days in program") +
  ylab("Age (years)") +
  xlim( c(-1, 300)) +
  ylim( c(-1, 60)) +
  geom_abline(intercept = linear_model_Days$coefficients[1], slope = linear_model_Days$coefficients[2])
# plot graph
#plot(plot_Days)

# set random generator seed
set.seed(7)

# get number of rows
n <- dim(data_clean)[1]

train_ind <- runif(n) < 0.75

# create test and train data-sets 
data_train <- data_clean[ train_ind, ]
data_test <- data_clean[ !train_ind, ]

#convert predictor columns to a factor
data_train$Race_American_Indian_or_Alaska_Native <- factor(data_train$Race_American_Indian_or_Alaska_Native)
data_test$Race_American_Indian_or_Alaska_Native <- factor(data_test$Race_American_Indian_or_Alaska_Native)
data_train$Race_White <- factor(data_train$Race_White)
data_test$Race_White <- factor(data_test$Race_White)
data_train$Race_Multi_Racial <- factor(data_train$Race_Multi_Racial)
data_test$Race_Multi_Racial <- factor(data_test$Race_Multi_Racial)
data_train$Race_Black_or_African_American <- factor(data_train$Race_Black_or_African_American)
data_test$Race_Black_or_African_American <- factor(data_test$Race_Black_or_African_American)
data_train$Race_Native_Hawaiian_or_Other_Pacific_Islander <- factor(data_train$Race_Native_Hawaiian_or_Other_Pacific_Islander)
data_test$Race_Native_Hawaiian_or_Other_Pacific_Islander <- factor(data_test$Race_Native_Hawaiian_or_Other_Pacific_Islander)

# convert target columns into factor
data_train$Race <- factor(data_train$Race)
data_test$Race <- factor(data_test$Race)

data_train$Ethnicity <- factor(data_train$Ethnicity)
data_test$Ethnicity <- factor(data_test$Ethnicity)

data_train$Gender <- factor(data_train$Gender)
data_test$Gender <- factor(data_test$Gender)


# Modeling using days at sheler as perdictor
c5_model_Race_days <- C5.0(formula = Race ~ Days_Enrolled, data = data_train)
c5_model_Gender_days <- C5.0(formula = Gender ~ Days_Enrolled, data = data_train)
c5_model_Ethnicity_days <- C5.0(formula = Ethnicity ~ Days_Enrolled, data = data_train)

data_train$C5_predicted_Race_days <- predict(object = c5_model_Race_days, newdata = data_train)
data_train$C5_predicted_Gender_days<- predict(object = c5_model_Gender_days, newdata = data_train)
data_train$C5_predicted_Ethnicity_days <- predict(object = c5_model_Ethnicity_days, newdata = data_train)

data_test$XT_Race_days <- subset(x=data_test, select = c("Days_Enrolled"))
data_test$XT_Gender_days <- subset(x=data_test, select = c("Days_Enrolled"))
data_test$XT_Ethnicity_days <- subset(x=data_test, select = c("Days_Enrolled"))

data_test$YTP_Race_days <- predict( object = c5_model_Race_days, newdata = data_test$XT_Race_days)
data_test$YTP_Gender_days <- predict( object = c5_model_Gender_days, newdata = data_test$XT_Gender_days)
data_test$YTP_Ethnicity_days <- predict( object = c5_model_Ethnicity_days, newdata = data_test$XT_Ethnicity_days)

crosstab_Race_days <- table (data_test$Race, data_test$YTP_Race_days)
crosstab_Gender_days <- table (data_test$Gender, data_test$YTP_Gender_days)
crosstab_Ethnicity_days <- table (data_test$Ethnicity, data_test$YTP_Ethnicity_days)

evals_Days <- model_eval(cross_tab = crosstab_Race_days, beta = 1, model_name = "C5 Model Race", binary = FALSE)
evals_Days <- cbind(evals_Days, model_eval(cross_tab = crosstab_Gender_days, beta = 1, model_name = "C5 Model Gender", binary = TRUE))
evals_Days <- cbind(evals_Days, model_eval(cross_tab = crosstab_Ethnicity_days, beta = 1, model_name = "C5 Model Ethnicity", binary = TRUE))


# Modeling using age at sheler as perdictor
c5_model_Race_age <- C5.0(formula = Race ~ Age, data = data_train)
c5_model_Gender_age <- C5.0(formula = Gender ~ Age, data = data_train)
c5_model_Ethnicity_age <- C5.0(formula = Ethnicity ~ Age, data = data_train)

data_train$C5_predicted_Race_age <- predict(object = c5_model_Race_age, newdata = data_train)
data_train$C5_predicted_Gender_age<- predict(object = c5_model_Gender_age, newdata = data_train)
data_train$C5_predicted_Ethnicity_age <- predict(object = c5_model_Ethnicity_age, newdata = data_train)

data_test$XT_Race_age <- subset(x=data_test, select = c("Age"))
data_test$XT_Gender_age <- subset(x=data_test, select = c("Age"))
data_test$XT_Ethnicity_age <- subset(x=data_test, select = c("Age"))

data_test$YTP_Race_age <- predict( object = c5_model_Race_age, newdata = data_test$XT_Race_age)
data_test$YTP_Gender_age <- predict( object = c5_model_Gender_age, newdata = data_test$XT_Gender_age)
data_test$YTP_Ethnicity_age <- predict( object = c5_model_Ethnicity_age, newdata = data_test$XT_Ethnicity_age)

crosstab_Race_age <- table (data_test$Race, data_test$YTP_Race_age)
crosstab_Gender_age <- table (data_test$Gender, data_test$YTP_Gender_age)
crosstab_Ethnicity_age <- table (data_test$Ethnicity, data_test$YTP_Ethnicity_age)

evals_Age <- model_eval(cross_tab = crosstab_Race_age, beta = 1, model_name = "C5 Model Race", binary = FALSE)
evals_Age <- cbind(evals_Age, model_eval(cross_tab = crosstab_Gender_age, beta = 1, model_name = "C5 Model Gender", binary = TRUE))
evals_Age <- cbind(evals_Age, model_eval(cross_tab = crosstab_Ethnicity_age, beta = 1, model_name = "C5 Model Ethnicity", binary = TRUE))


# Modeling using age & days at sheler as perdictor
c5_model_Race_days_age <- C5.0(formula = Race ~ Days_Enrolled + Age, data = data_train)
c5_model_Gender_days_age <- C5.0(formula = Gender ~ Days_Enrolled + Age, data = data_train)
c5_model_Ethnicity_days_age <- C5.0(formula = Ethnicity ~ Days_Enrolled + Age, data = data_train)

data_train$C5_predicted_Race_days_age <- predict(object = c5_model_Race_days_age, newdata = data_train)
data_train$C5_predicted_Gender_days_age<- predict(object = c5_model_Gender_days_age, newdata = data_train)
data_train$C5_predicted_Ethnicity_days_age <- predict(object = c5_model_Ethnicity_days_age, newdata = data_train)

data_test$XT_Race_days_age <- subset(x=data_test, select = c("Days_Enrolled", "Age"))
data_test$XT_Gender_days_age <- subset(x=data_test, select = c("Days_Enrolled", "Age"))
data_test$XT_Ethnicity_days_age <- subset(x=data_test, select = c("Days_Enrolled", "Age"))

data_test$YTP_Race_days_age <- predict( object = c5_model_Race_days_age, newdata = data_test$XT_Race_days_age)
data_test$YTP_Gender_days_age <- predict( object = c5_model_Gender_days_age, newdata = data_test$XT_Gender_days_age)
data_test$YTP_Ethnicity_days_age <- predict( object = c5_model_Ethnicity_days_age, newdata = data_test$XT_Ethnicity_days_age)

crosstab_Race_days_age <- table (data_test$Race, data_test$YTP_Race_days_age)
crosstab_Gender_days_age <- table (data_test$Gender, data_test$YTP_Gender_days_age)
crosstab_Ethnicity_days_age <- table (data_test$Ethnicity, data_test$YTP_Ethnicity_days_age)

evals_Days_Age <- model_eval(cross_tab = crosstab_Race_days_age, beta = 1, model_name = "C5 Model Race", binary = FALSE)
evals_Days_Age <- cbind(evals_Days_Age, model_eval(cross_tab = crosstab_Gender_days_age, beta = 1, model_name = "C5 Model Gender", binary = TRUE))
evals_Days_Age <- cbind(evals_Days_Age, model_eval(cross_tab = crosstab_Ethnicity_days_age, beta = 1, model_name = "C5 Model Ethnicity", binary = TRUE))