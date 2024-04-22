## Plotting Scatter plor matrix
# Load dataset from a CSV file
plot_data <- read.csv("Dataset/medical_insurance_modified_2.csv")
pairs(plot_data)

## Regression
# Load dataset from a CSV file
data <- read.csv("Dataset/medical_insurance_modified_1.csv")

# Printing Column names
colnames(data)

## Multicollinearity Check by Variance Inflation Factor
# Fitting Linear model
lm_model <- lm(charges ~ age + bmi + children + sex + smoker + region1 + region2 + region3 , data=data)
library(carData)
library(car)
summary(lm_model)

# removing sex, region1 from model at 1% level of significance
vif(lm_model)

## Best linear regression
# By stepwise regression selected variables are- age, bmi, children, smoker, region2, region3 
lm_model <- lm(charges ~ age + bmi + children  + smoker + region2 + region3 , data=data)
summary(lm_model)

# Predict values using the model
predicted_values <- predict(lm_model)

# Calculate absolute differences
abs_diff <- abs(data$charges - predicted_values)

# Calculate Mean Absolute Error (MAE)
mae_lm <- mean(abs_diff)

# Print MAE
print(mae_lm)

# ANOVA
anova(lm_model)

# Confidence interval for coefficients
confint(lm_model,level=0.95)

# Extract studentized residuals
studentized_res <- rstudent(lm_model)

# Plot studentized residuals
fit = fitted(lm_model)
plot(fit, studentized_res, xlab = "fit", ylab = "Studentized Residuals")
abline(0,0)

# Create Normal probability plot of residuals
qqnorm(studentized_res,xlab="Normal Scores",ylab = "Studentized Residuals")
qqline(studentized_res)

# Plot histogram of studentized residuals
hist(studentized_res, main = "Distribution of Residuals", xlab = "Residuals", ylab = "Frequency", col = "skyblue")

# Testing for normality of studentized residuals
shapiro.test(studentized_res)
ks_result <- ks.test(studentized_res, "pnorm")
ks_result

## Influential Point Checking
# Calculate Cook's distance
cooksd <- cooks.distance(lm_model)

# Plot Cook's distance
plot(fit,cooksd, pch = 20, xlab = "Observation", ylab = "Cook's Distance")
abline(h = 4/length(cooksd), col = "red")

# 10 fold CV
#install.packages("caret")
#install.packages("ggplot2")
#install.packages("lattice")

# Load necessary libraries
library(lattice)
library(ggplot2)
library(caret)

# Set up your control parameters for 10-fold cross-validation
ctrl <- trainControl(method = "cv", 
                     number = 10, # Number of folds
                     summaryFunction = defaultSummary, # Use default summary function for regression
                     verboseIter = TRUE) # To display progress

# Define your linear regression model
model <- train(charges ~ age + bmi + children  + smoker + region2 + region3 , 
               data = data, 
               method = "lm", # Linear regression method
               trControl = ctrl)

# Print the results
print(model)


## Polynomial Regression

degree <- 2

poly_model <- lm(charges ~ poly(age,degree) + poly(bmi,degree) + poly(children,degree)  + age:bmi + age:smoker + bmi:smoker + smoker  + region2:age + region2:bmi + region2  + region3:age + region3:bmi + region3 , data=data)
summary(poly_model)

# Deleting variables with low p value
poly_model <- lm(charges ~ polym(age, bmi, children, smoker,region2, region3, degree=2, raw=TRUE),data=data)
summary(poly_model)

# Deleting variables with low p value
poly_model <- lm(charges ~ I(age^2)+ age + I(bmi^2) + children + bmi:smoker + smoker  + region2:age + region2:bmi  , data=data)
summary(poly_model)

# Best One
poly_model <- lm(charges ~ I(age^2) + children + bmi:smoker + smoker  + region2:age + region2:bmi  , data=data)
summary(poly_model)

# MAE
predicted_values <- predict(poly_model)
abs_diff <- abs(data$charges - predicted_values)
mae <- mean(abs_diff)
mae

# ANOVA
anova(poly_model)

# Confidence interval for coefficients
confint(poly_model,level=0.95)

# Extract studentized residuals
studentized_res <- rstudent(poly_model)

# Plot studentized residuals
fit = fitted(poly_model)
plot(fit, studentized_res, xlab = "fit", ylab = "Studentized Residuals")
abline(0,0)

# Create Normal probability plot of residuals
qqnorm(studentized_res,xlab="Normal Scores",ylab = "Studentized Residuals")
qqline(studentized_res)

# Plot histogram of studentized residuals
hist(studentized_res, main = "Distribution of Residuals", xlab = "Residuals", ylab = "Frequency", col = "skyblue")

# Testing for normality of studentized residuals
shapiro.test(studentized_res)
ks<-ks.test(studentized_res, "pnorm")
ks

# 10-fold CV
model <- train(charges ~ I(age^2) + children + bmi:smoker + smoker  + region2:age + region2:bmi  ,
               data = data, 
               method = "lm", # Linear regression method
               trControl = ctrl)
print(model)

## RIDGE REGRESSION

#install.packages("glmnet")
library(glmnet)

# Processing dataset
data$age2 <- data$age * data$age
data <- subset(data, select = -age)
data$bmi.smoker <- data$bmi * data$smoker
data$region2.age <- data$age * data$region2
data$region2.bmi <- data$bmi * data$region2
data <- subset(data, select = -bmi)
data <- subset(data, select = -sex)
data <- subset(data, select = -region1)
data <- subset(data, select = -region2)
data <- subset(data, select = -region3)

# Design and target matrix
X <- as.matrix(data[, -1])
y <- data[, 1]

# Model Building
ridge_model <- cv.glmnet(X, y, alpha = 0)

# Optimal lambda
print(ridge_model$lambda.min)

# Coefficients
coef(ridge_model, s = "lambda.min")
summary(ridge_model)

# Prediction
y_pred <- predict(ridge_model, X, s = "lambda.min")  # Use the lambda value that minimizes the cross-validation error

# Calculate R-squared score, MSE and MAE
r2_score <- R2(y, y_pred)
mse <- mean((y - y_pred)^2)
mae <- mean(abs(y - y_pred))
cat("R2 Score:", r2_score, "\n")
cat("Mean Squared Error:", mse, "\n")
cat("Mean Absolute Error:", mae, "\n")

# 10-fold CV
model <- train(charges ~ .,                   # Formula specifying the outcome variable and predictors
               data = data,      
               method = "ridge",          
               trControl = ctrl
)       # Specify the control parameters

# Print the cross-validation results
print(model)

## LASSO regression

# Select the optimal lambda using cross-validation
cv_lasso <- cv.glmnet(x = X, y = y, alpha = 1)
max(cv_lasso$cvm)

# Get the optimal lambda
optimal_lambda <- cv_lasso$lambda.min
cat("Optimal lambda:", optimal_lambda, "\n")

# Plotting lambda vs error
plot(cv_lasso) 

# Best model
best_model <- glmnet(X, y, alpha = 1, lambda = optimal_lambda)
coef(best_model)

# use fitted best model to make predictions
y_predicted <- predict(best_model, s = optimal_lambda, newx = X)

# find SST and SSE
sst <- sum((y - mean(y))^2)
sse <- sum((y_predicted - y)^2)

# find R-Squared
rsq <- 1 - sse/sst
rsq

# Compute Mean Squared Error
mse <- mean((y - y_predicted)^2)
cat("Mean Squared Error:", mse, "\n")

# Compute Mean Absolute Error
mae <- mean(abs(y - y_predicted))
cat("Mean Absolute Error:", mae, "\n")

# 10-fold CV
model <- train(charges ~ .,                   # Formula specifying the outcome variable and predictors
               data = data,      
               method = "lasso",          
               trControl = ctrl
)       # Specify the control parameters

# Print the cross-validation results
print(model)


## Random Forest Regression

#install.packages("randomForest") 

library(randomForest) 
set.seed(4543)

# Fitting Random Forest model
rf.fit <- randomForest(charges ~ ., data=data, mtry=4,
                       keep.forest=TRUE, importance=TRUE) 
rf.fit 
predicted_values <- predict(rf.fit)

# Calculate absolute differences
abs_diff <- abs(data$charges - predicted_values)

# Calculate Mean Absolute Error (MAE)
mae <- mean(abs_diff)

# Print MAE
print(mae)

# Plotting error vs number of trees
plot(rf.fit)

# 10-fold CV
model <- train(charges ~ .,                   # Formula specifying the outcome variable and predictors
               data = data,       
               method = "rf",         
               trControl = ctrl
               )       # Specify the control parameters

# Print the cross-validation results
print(model)