##################################
# Group Name : 
# 
#
#
#
##################################

# set working directory
setwd("/Users/jolde/Desktop/CS File/CS-251/Final")

# Read Data into the Program. 
data <- read.csv("initial_data.csv")

# replace empty slots with NA
data[data==""] <- NA

# Convert Truth Values to 1 or 0.
data[data=="Yes"] <- 1
data[data=="No"] <- 0
data[data=="True"] <- 1
data[data=="False"] <- 0 

# remove redacted columns
data$X3.10.Enroll.Date <- NULL
data$X3.11.Exit.Date <- NULL
data$Date.of.Last.Contact..Beta. <- NULL
data$Date.of.First.Contact..Beta. <- NULL
data$Date.of.Last.ES.Stay..Beta. <- NULL
data$Date.of.First.ES.Stay..Beta. <- NULL
data$CurrentDate <- NULL
data$X3.917.Homeless.Start.Date <- NULL

# remove empty columns
data$Latitude <- NULL
data$Longitude <- NULL
data$R1.Referral.Source <- NULL
data$R2.Date.Status.Determined <- NULL
data$R2.Enroll.Status <- NULL
data$R2.Runaway.Youth <- NULL
data$R2.Reason.Why.No.Services.Funded <- NULL
data$R3.Sexual.Orientation <- NULL
data$R6.Employed.Status <- NULL
data$R6.Why.Not.Employed <- NULL
data$R6.Type.of.Employment <- NULL
data$R6.Looking.for.Work <- NULL
data$R7.General.Health.Status <- NULL
data$R8.Dental.Health.Status <- NULL
data$R9.Mental.Health.Status <- NULL
data$R10.Pregnancy.Status <- NULL
data$R10.Pregnancy.Due.Date <- NULL
data$X4.02.Total.Income.at.Annual.Update <- NULL
data$SOAR.Eligibility.Determination..Most.Recent.<- NULL
data$SOAR.Enrollment.Determination..Most.Recent.<- NULL
data$PSH...Most.Recent.Enrollment <- NULL
data$X4.08.HIV.AIDS <- NULL
data$HEN.HP.Referral.Most.Recent <- NULL
data$HEN.RRH.Referral.Most.Recent <- NULL
data$WorkSource.Referral.Most.Recent <- NULL
data$YAHP.Referral.Most.Recent <- NULL
data$X3.917b.Stayed.Less.Than.7.Nights <- NULL
data$X3.917.Stayed.Less.Than.90.Days <- NULL
data$X3.917b.Stayed.in.Streets..ES.or.SH.Night.Before <- NULL
data$Municipality..City.or.County. <- NULL
data$RRH.In.Permanent.Housing <- NULL
data$RRH.Date.Of.Move.In <- NULL
data$X4.13.Engagement.Date <- NULL
data$X4.24.In.School..Retired.Data.Element. <- NULL
data$X2.1.Organization.Name <- NULL

# remove columns containing the same information
data$X2.2.Project.Name <- NULL
data$X2.4.ProjectType <- NULL
data$X2.5.Utilization.Tracking.Method..Invalid. <- NULL
data$X2.6.Federal.Grant.Programs <- NULL

# export data frame
write.csv(data, "/Users/jolde/Desktop/CS File/CS-251/Final/data_cleaned.csv", row.names = FALSE)

