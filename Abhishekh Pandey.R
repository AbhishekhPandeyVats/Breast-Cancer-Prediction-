library(boot)
library(randomForest)
library(neuralnet)
library(ggplot2)
library(reshape2)
library(car)
library(caret)
library(data.table)
library(caTools)
library(earth)

set.seed(1)

setwd('C:/Users/abhis/OneDrive/Desktop/Academics/Year 2 Sem 2/BC2407 Analytic II Advanced Predictive Techniques/AY22s2 BC2407 CBA')

original.dt<- read.csv('wisconsin breast cancer data.csv')

#creating a cleaned dataset 
clean.dt <- copy(original.dt)

#dropping id column 
clean.dt$id <- NULL

#converting diagnosis variable into categorical variable
clean.dt$diagnosis <- factor(clean.dt$diagnosis)

#creating a correlation matrix using Pearson Correlation
correlation.dt <- copy(clean.dt)
correlation.dt$diagnosis <- NULL
corr_matrix <- cor(correlation.dt, method = "pearson")
print(corr_matrix)

# Filter out values greater than 0.8
corr_matrix[abs(corr_matrix) <= 0.8] <- NA

melted_corr <- melt(corr_matrix)

#Plotting the correlation matrix into a heatmap
ggplot(melted_corr, aes(x = Var2, y = Var1, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "#0000FF", high = "#FF0000") +
  theme_minimal() +
  labs(title = "Pearson Correlation Matrix") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        axis.text = element_text(size = 10),
        axis.title = element_text(size = 12),
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 10)) +
  geom_text(aes(label = round(value, 2)), size = 2, color = "black")

drop <- c('perimeter_mean', 'perimeter_worst', 'perimeter_se', 'area_mean', 'area_worst', 'area_se')
clean.dt<- clean.dt[, !(names(clean.dt) %in% drop)]

keep_cols <- c('texture_mean', 'radius_mean', 'concavity_mean', 'fractal_dimension_mean', 'radius_se', 'smoothness_worst',
               'compactness_worst', 'symmetry_worst', 'diagnosis')

drop_cols <- setdiff(names(clean.dt), keep_cols)

clean.dt<- clean.dt[, !(names(clean.dt) %in% drop_cols)]

#creating a correlation matrix using Pearson Correlation
correlation.dt <- copy(clean.dt)
correlation.dt$diagnosis <- NULL
corr_matrix <- cor(correlation.dt, method = "pearson")
print(corr_matrix)

# Filter out values greater than 0.7
corr_matrix[abs(corr_matrix) <= 0.7] <- NA

# Plot the filtered correlation matrix
melted_corr <- melt(corr_matrix)
ggplot(melted_corr, aes(x = Var2, y = Var1, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "#0000FF", high = "#FF0000") +
  theme_minimal() +
  labs(title = "Pearson Correlation Matrix (Filtered)") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        axis.text = element_text(size = 10),
        axis.title = element_text(size = 12),
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 10)) +
  geom_text(aes(label = round(value, 2)), size = 2, color = "black")

n_pos <- sum(clean.dt$diagnosis == 'B')

# Set the proportion of positive samples to keep
p_keep <- 0.6

# Calculate the number of positive samples to keep
n_pos_keep <- round(n_pos * p_keep)

# Create a vector of positive sample indices
pos_indices <- which(clean.dt$diagnosis == 'B')

# Randomly select the positive sample indices to keep
pos_indices_keep <- sample(pos_indices, n_pos_keep)

# Create a vector of all sample indices to keep
indices_keep <- c(pos_indices_keep, which(clean.dt$diagnosis == 'M'))

# Subset the dataframe to keep the selected rows
clean.dt <- clean.dt[indices_keep,]

summary(clean.dt)

train <- sample.split(Y = clean.dt$diagnosis, SplitRatio = 0.7)
trainset <- subset(clean.dt, train == T)
testset <- subset(clean.dt, train == F)


#==========================================================================================
#------------------------------ Exploratory Data Analysis ------------------------------
#==========================================================================================
ggplot(data=clean.dt, aes(x=symmetry_worst, y=diagnosis, fill=diagnosis)) + 
  geom_boxplot() +
  scale_fill_manual(values = c("B" = "light blue", "M" = "orange")) +
  labs(title = "Distribution of Diagnosis vs. symmetry_worst")

ggplot(data=clean.dt, aes(x=compactness_worst, y=diagnosis, fill=diagnosis)) + 
  geom_boxplot() +
  scale_fill_manual(values = c("B" = "light blue", "M" = "orange")) +
  labs(title = "Distribution of Diagnosis vs. compactness_worst")

ggplot(data=clean.dt, aes(x=smoothness_worst, y=diagnosis, fill=diagnosis)) + 
  geom_boxplot() +
  scale_fill_manual(values = c("B" = "light blue", "M" = "orange")) +
  labs(title = "Distribution of Diagnosis vs. smoothness_worst")

ggplot(data=clean.dt, aes(x=radius_se, y=diagnosis, fill=diagnosis)) + 
  geom_boxplot() +
  scale_fill_manual(values = c("B" = "light blue", "M" = "orange")) +
  labs(title = "Distribution of Diagnosis vs. radius_se")

ggplot(data=clean.dt, aes(x=fractal_dimension_mean, y=diagnosis, fill=diagnosis)) + 
  geom_boxplot() +
  scale_fill_manual(values = c("B" = "light blue", "M" = "orange")) +
  labs(title = "Distribution of Diagnosis vs. fractal_dimension_mean")

ggplot(data=clean.dt, aes(x=concavity_mean, y=diagnosis, fill=diagnosis)) + 
  geom_boxplot() +
  scale_fill_manual(values = c("B" = "light blue", "M" = "orange")) +
  labs(title = "Distribution of Diagnosis vs. concavity_mean")

ggplot(data=clean.dt, aes(x=texture_mean, y=diagnosis, fill=diagnosis)) + 
  geom_boxplot() +
  scale_fill_manual(values = c("B" = "light blue", "M" = "orange")) +
  labs(title = "Distribution of Diagnosis vs. texture_mean")

ggplot(data=clean.dt, aes(x=radius_mean, y=diagnosis, fill=diagnosis)) + 
  geom_boxplot() +
  scale_fill_manual(values = c("B" = "light blue", "M" = "orange")) +
  labs(title = "Distribution of Diagnosis vs. radius_mean")

ggplot(data=clean.dt, aes(x = compactness_worst, y = concavity_mean)) +
  geom_point(color= "steelblue") + geom_smooth(formula = y ~ x,method = "lm") +
  ggtitle("Scatterplot of compactness_worst against concavity_mean")


#==========================================================================================
#------------------------------ Model 1: Random Forest Model ------------------------------
#==========================================================================================

rf <-randomForest(diagnosis~texture_mean + radius_mean + concavity_mean + fractal_dimension_mean + radius_se + smoothness_worst +
                  compactness_worst + symmetry_worst,data=trainset, ntree=500)
print(rf)

#Find optimal mtry value
#Step Factor: To test for different mtry values by scaling by this value
#Improve: Required improvement to continue testing for other mtry
#Trace: To print the progress of search
#Plot: To plot the OOB error as function of mtry
mtry <- tuneRF(trainset[,1:(length(trainset)-1)],trainset$diagnosis, ntreeTry=500,
               stepFactor=1,improve=0.01, trace=TRUE, plot=TRUE)

#Find mtry with lowest OOBError
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m) #mtry 2 is the best mtry

#Build model again with best mtry value
rf2 <- randomForest(diagnosis~texture_mean + radius_mean + concavity_mean + fractal_dimension_mean + radius_se + smoothness_worst +
                    compactness_worst + symmetry_worst,data=trainset, mtry=best.m, importance=TRUE, ntree=500)
print(rf2)

#Evaluate variable importance
importance(rf2)
varImpPlot(rf2)

#Predict on testset
predictions.rf = predict(rf2, newdata=testset, type="class")

#All in one confusion matrix
confusionMatrix(predictions.rf, reference = testset$diagnosis, positive="M", mode="everything")

#==========================================================================================
#------------------------------------- Model 2: Logistic Regression -------------------------------------
#==========================================================================================

trainset$diagnosis2 <- ifelse(trainset$diagnosis == "M", 1, 0)
trainset$diagnosis2 <- factor(trainset$diagnosis2)
summary(trainset)
testset$diagnosis2 <- ifelse(testset$diagnosis == "M", 1, 0)
testset$diagnosis2 <- factor(testset$diagnosis2)
summary(testset)

#Building model first time
m.logisticregression <- glm(diagnosis2 ~ texture_mean + radius_mean + concavity_mean + fractal_dimension_mean + radius_se + smoothness_worst +
                            compactness_worst + symmetry_worst, family = binomial, data = trainset)

#Building model second time, remove insignificant variables
m.logisticregression = step(m.logisticregression)
summary(m.logisticregression)

#odd ratio
OR <- exp(coef(m.logisticregression))
OR

#Find the confidence interval
OR.CI <- exp(confint(m.logisticregression))
OR.CI

# Predicting on Testset
predictions.logisticregression <- predict(m.logisticregression, newdata = testset, type = 'response')
predictions.logisticregression  <- ifelse(predictions.logisticregression >0.5, 1, 0)
predictions.logisticregression <- factor(predictions.logisticregression)

##-----Evaluation of the model-----##
# Quick overview of model
confusionMatrix(predictions.logisticregression, reference = testset$diagnosis2, positive='1', mode="everything")
  
#==========================================================================================
#------------------------------------- Model 3: MARS -------------------------------------
#==========================================================================================
m.mars <- earth(diagnosis ~ texture_mean + radius_mean + concavity_mean + fractal_dimension_mean + radius_se + smoothness_worst +
                  compactness_worst + symmetry_worst, data=trainset, degree=1)
summary(m.mars)
evimp(m.mars)

predictions.mars <- predict(m.mars, newdata=testset, type="response")
predictions.mars  <- ifelse(predictions.mars >0.5, 'M', 'B')
predictions.mars <- factor(predictions.mars)

confusionMatrix(table(predictions.mars, testset$diagnosis), reference = testset$diagnosis, positive="M", mode="everything")
