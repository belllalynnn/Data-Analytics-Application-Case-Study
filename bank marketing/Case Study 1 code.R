#######################################################################
###########################  CASE STUDY 1  ############################
#######################################################################

library(MASS) ; library(ggplot2) ; library(tidyverse) ;library(corrplot)
library(corrplot) ; library(car) ; library(caret)

setwd("/Users/BellaIsCute/Desktop/DA Application") # Change this to your location :)

# Loading data
bank <- read.csv('bank-additional.csv', na.strings = "unknown", sep =';')
bank <- na.omit(bank)

# Take a look at the data
names(bank)
str(bank)
levels(bank$y) # -> Classification problem

# Changing to factor
bank$job = as.factor(bank$job)
bank$marital = as.factor(bank$marital)
bank$education = as.factor(bank$education)
bank$default = as.factor(bank$default)
bank$housing = as.factor(bank$housing)
bank$loan = as.factor(bank$loan)
bank$contact = as.factor(bank$contact)
bank$month = as.factor(bank$month)
bank$day_of_week = as.factor(bank$day_of_week)
bank$poutcome = as.factor(bank$poutcome)
bank$y = as.factor(bank$y)

bank$pdays <- cut(bank$pdays, c(0, 7, 14, 998),
                              c("Less than 1 week", "Between 1-2 weeks",
                                "More than 2 weeks"), right = TRUE)

bank$contacted <- ifelse(is.na(bank$pdays), "No", "Yes")
bank$contacted <- as.factor(bank$contacted)

summary(bank)
# Removing 'default' because only one "yes" and removing duration
bank = dplyr::select(bank, -default)
bank = dplyr::select(bank, -duration)

# Looking at the relationships between the numeric variables
bank_num <- dplyr::select_if(bank, is.numeric)
M = cor(bank_num)
corrplot(M, method = c("number"))

pairs(bank_num)
summary(bank_num)

# Dummy Variables (marital, housing, loan, contact, poutcome, contacted)
bank$marital_dummy1 <- ifelse(bank$marital == "divorced", 1, 0)
bank$marital_dummy2 <- ifelse(bank$marital == "married", 1, 0)

bank$housing_dummy <- ifelse(bank$housing == "yes", 1, 0)

bank$loan_dummy <- ifelse(bank$loan == "yes", 1, 0)

bank$contact_dummy <- ifelse(bank$contact == "cellular", 1, 0)

bank$poutcome_dummy1 <- ifelse(bank$poutcome == "failure", 1, 0)
bank$poutcome_dummy2 <- ifelse(bank$poutcome == "success", 1, 0)

bank$contacted_dummy <- ifelse(bank$contacted == "Yes", 1, 0)


### Splitting into test and train
set.seed(1)
index = sample(nrow(bank), 0.8*nrow(bank), replace = F) # 80/20 split
bank_train = bank[index,]
bank_test = bank[-index,]


      ############# Logistic Regression: UNBALANCED Data ##############

### Building model with unbalanced data
m1 <- glm(y ~ age+marital_dummy1+marital_dummy2+housing_dummy+loan_dummy+
            contact_dummy+campaign+previous+poutcome_dummy1+poutcome_dummy2+
            emp.var.rate+cons.price.idx+cons.conf.idx+euribor3m+nr.employed+
            contacted_dummy, data = bank_train, family = binomial)
summary(m1)

### Checking for multicolliniarity
vif(m1) 
# Removing euribor3m
m1 <- glm(y ~ age+marital_dummy1+marital_dummy2+housing_dummy+loan_dummy+
            contact_dummy+campaign+previous+poutcome_dummy1+poutcome_dummy2+
            emp.var.rate+cons.price.idx+cons.conf.idx+nr.employed+contacted_dummy,
          data = bank_train, family = binomial)
vif(m1)
# Removing emp.var.rate
m1 <- glm(y ~ age+marital_dummy1+marital_dummy2+housing_dummy+loan_dummy+
            contact_dummy+campaign+previous+poutcome_dummy1+poutcome_dummy2+
            cons.price.idx+cons.conf.idx+nr.employed+contacted_dummy,
          data = bank_train, family = binomial)
vif(m1)
# Removing contacted_dummy
m1 <- glm(y ~ age+marital_dummy1+marital_dummy2+housing_dummy+loan_dummy+
            contact_dummy+campaign+previous+poutcome_dummy1+poutcome_dummy2+
            cons.price.idx+cons.conf.idx+nr.employed,
          data = bank_train, family = binomial)
vif(m1)

### Diagnostic plots

# Linearity diagnostic plot using Pearson residuals
resid <- residuals(m1, type = "deviance")
std_res <- resid / sqrt(1 - hatvalues(m1))

par(mfrow = c(1, 1))
plot(std_res, pch = 20, col = ifelse(bank_train$y == "no", "#dac1ac", "#6c8ca0"),
     ylim = c(-3.5, 3.5), ylab = "Standardized Deviance Residuals", xlab = "ID")

# Influential points using Cook's distance
cook_d <- cooks.distance(m1)
inf_id <- which(cook_d > 0.4)

par(mfrow = c(1, 1))
plot(cook_d, pch = 20, col = ifelse(bank_train$y == "no", "#dac1ac", "#6c8ca0"),
     ylim = c(0, max(cook_d) * 1.2), ylab = "Cook's Distance", xlab = "ID")

# Which observation has cook's d larger than 0.25?
(inf.id=which(cooks.distance(m1)>0.25)) # No influential points


##### Predicting the responses on m1 unbalanced testing data. 
predprob_log <- predict.glm(m1, bank_test, type = "response")  ## for logit
predclass_log = ifelse(predprob_log >= 0.5, "yes", "no")

# Confusion matrix
caret::confusionMatrix(as.factor(predclass_log), bank_test$y, positive = "yes")
# Accuracy    : 0.8964
# Sensitivity : 0.23377          
# Specificity : 0.99076 


#####  Stepwise Selection with AIC
null_model = glm(y ~ 1, data = bank, family = binomial) 
full_model = m1 

step.model.AIC = step(null_model, scope = list(upper = full_model),
                      direction = "both", test = "Chisq", trace = F) 
summary(step.model.AIC) 

# Best model based on stepwise selection
m2 <- glm(y ~ contact_dummy+campaign+poutcome_dummy1+poutcome_dummy2+
            cons.price.idx+cons.conf.idx+nr.employed, bank_train, family = binomial)
summary(m2)


### Diagnostic plots

# Linearity diagnostic plot using Pearson residuals
resid <- residuals(m2, type = "deviance")
std_res <- resid / sqrt(1 - hatvalues(m1))

par(mfrow = c(1, 1))
plot(std_res, pch = 20, col = ifelse(bank_train$y == "no", "#dac1ac", "#6c8ca0"),
     ylim = c(-3.5, 3.5), ylab = "Standardized Deviance Residuals", xlab = "ID")

# Influential points using Cook's distance
cook_d <- cooks.distance(m2)
inf_id <- which(cook_d > 0.4)

par(mfrow = c(1, 1))
plot(cook_d, pch = 20, col = ifelse(bank_train$y == "no", "#dac1ac", "#6c8ca0"),
     ylim = c(0, max(cook_d) * 1.2), ylab = "Cook's Distance", xlab = "ID")

# Which observation has cook's d larger than 0.25?
(inf.id=which(cooks.distance(m1)>0.25)) # No influential points


##### Predict the responses on the testing data
bank_test$PredProb = predict.glm(m2, newdata = bank_test, type = "response") 
bank_test$PredY = ifelse(bank_test$PredProb >= 0.5, "yes", "no")

caret::confusionMatrix(as.factor(bank_test$y), as.factor(bank_test$PredY), positive = "yes")
# Accuracy    : 0.8932
# Sensitivity : 0.72000         
# Specificity : 0.90051



        ############# Logistic Regression: BALANCED Data ##############

### Resample with more balanced data 
bank_yes_cust = bank %>% filter(y == "yes")
bank_no_cust = bank %>% filter(y == "no")

set.seed(1)
sample_yes_cust = sample_n(bank_yes_cust, nrow(bank_no_cust), replace = TRUE) #matching the same row number
bank_bal = rbind(bank_no_cust,sample_yes_cust) #combining both so they will have a balance data


# Split data into training and testing balanced samples
set.seed(1) 
index_bal <- sample(nrow(bank_bal),0.8*nrow(bank_bal),replace = F) # 80/20 split
banktrain_bal <- bank_bal[index_bal,]
banktest_bal <- bank_bal[-index_bal,]


### Building model with balanced data
m1_bal = glm(y ~ age + marital_dummy1 + marital_dummy2 + housing_dummy + loan_dummy + 
               contact_dummy + campaign + previous + poutcome_dummy1 + poutcome_dummy2 +
               cons.price.idx + cons.conf.idx  + nr.employed, data = banktrain_bal,
             family = binomial)
summary(m1_bal) # Look at results

vif(m1_bal) # Double checking multicollinearity -> Nothing to remove

### Diagnostic plots

# Linearity diagnostic plot using Pearson residuals
resid <- residuals(m1_bal, type = "deviance")
std_res <- resid / sqrt(1 - hatvalues(m1))

par(mfrow = c(1, 1))
plot(std_res, pch = 20, col = ifelse(bank_train$y == "no", "#dac1ac", "#6c8ca0"),
     ylim = c(-3.5, 3.5), ylab = "Standardized Deviance Residuals", xlab = "ID")

# Influential points using Cook's distance
cook_d <- cooks.distance(m1_bal)
inf_id <- which(cook_d > 0.4)

par(mfrow = c(1, 1))
plot(cook_d, pch = 20, col = ifelse(bank_train$y == "no", "#dac1ac", "#6c8ca0"),
     ylim = c(0, max(cook_d) * 1.2), ylab = "Cook's Distance", xlab = "ID")

# Which observation has cook's d larger than 0.25?
(inf.id=which(cooks.distance(m1)>0.25)) # No influential points


##### Predicting the responses on the balanced testing data. 
predprob_log_bal <- predict.glm(m1_bal, banktest_bal, type = "response")
predclass_log_bal = ifelse(predprob_log_bal >= 0.5, "yes", "no")

# Comparing m1 and m1_bal using the confusion matrix. 
caret::confusionMatrix(as.factor(predclass_log), bank_test$y,
                       positive = "yes") # m1 unbalanced (results on line 113)
caret::confusionMatrix(as.factor(predclass_log_bal), banktest_bal$y,
                       positive = "yes") # m1 balanced
# Accuracy    : 0.7518
# Sensitivity : 0.7893         
# Specificity : 0.7893


##### Using predictors from m2 on balanced data
# (because stepwise selection kept all predictors from m1_bal)
m2_bal = glm(y ~ nr.employed + contact_dummy + poutcome_dummy1 + cons.conf.idx
                  + campaign + cons.price.idx + poutcome_dummy2,
                  data = banktrain_bal, family = binomial) 
summary(m2_bal) # Look at results

vif(m2_bal) # Double check multicollinearity -> Nothing to remove


### Diagnostic plots

# Linearity diagnostic plot using Pearson residuals
resid <- residuals(m2_bal, type = "deviance")
std_res <- resid / sqrt(1 - hatvalues(m1))

par(mfrow = c(1, 1))
plot(std_res, pch = 20, col = ifelse(bank_train$y == "no", "#dac1ac", "#6c8ca0"),
     ylim = c(-3.5, 3.5), ylab = "Standardized Deviance Residuals", xlab = "ID")

# Influential points using Cook's distance
cook_d <- cooks.distance(m2_bal)
inf_id <- which(cook_d > 0.4)

par(mfrow = c(1, 1))
plot(cook_d, pch = 20, col = ifelse(bank_train$y == "no", "#dac1ac", "#6c8ca0"),
     ylim = c(0, max(cook_d) * 1.2), ylab = "Cook's Distance", xlab = "ID")

# Which observation has cook's d larger than 0.25?
(inf.id=which(cooks.distance(m1)>0.25)) # No influential points


###### Predict the responses on the testing data. 
predprob2_log_bal <- predict.glm(m2_bal, banktest_bal, type = "response")  
predclass2_log_bal = ifelse(predprob2_log_bal >= 0.5, "yes", "no")

# Confusion matrix
caret::confusionMatrix(as.factor(bank_test$y), as.factor(bank_test$PredY),
                       positive = "yes") # m2 unbalanced (results on line 136)
caret::confusionMatrix(as.factor(predclass2_log_bal), banktest_bal$y,
                       positive = "yes") # m2 balanced
# Accuracy    : 0.7583
# Sensitivity : 0.8089          
# Specificity : 0.7045 




      ################ Logistic Discriminant Analysis ################ 

##### Using UNBALANCED data

# LDA modeling
lda.model = lda(y ~ age+marital_dummy1+marital_dummy2+housing_dummy+loan_dummy+
                  contact_dummy+campaign+previous+poutcome_dummy1+poutcome_dummy2+
                  emp.var.rate+cons.price.idx+cons.conf.idx+euribor3m+nr.employed+
                  contacted_dummy, data = bank_train)

# View the output
lda.model

# Predicting for the testing dataset we created
predictions.lda = predict(lda.model, bank_test)

# Make confusion matrix for the LDA predictions to compare accuracy 
caret::confusionMatrix(as.factor(predictions.lda$class), bank_test$y)
# Accuracy    : 0.8851
# Sensitivity : 0.9649          
# Specificity : 0.3247


##### Using BALANCED data

# LDA modeling
lda.model_bal = lda(y ~ age+marital_dummy1+marital_dummy2+housing_dummy+loan_dummy+
                  contact_dummy+campaign+previous+poutcome_dummy1+poutcome_dummy2+
                  emp.var.rate+cons.price.idx+cons.conf.idx+euribor3m+nr.employed+
                  contacted_dummy, data = banktrain_bal)

# View the output
lda.model_bal

# Predicting for the testing dataset we created
predictions.lda_bal = predict(lda.model_bal, banktest_bal)

# Make confusion matrix for the LDA predictions to compare accuracy 
caret::confusionMatrix(as.factor(predictions.lda_bal$class), banktest_bal$y)
# Accuracy    : 0.7583
# Sensitivity : 0.8089          
# Specificity : 0.7045 



            ############### Evaluating Models ###############

models <- c("m1", "m2", "m1_bal", "m2_bal", "lda", "lda_bal")
Accuracy <- c(0.8964, 0.8932, 0.7518, 0.7583, 0.8851, 0.7583)
Sensitivity <- c(0.23377, 0.72, 0.7893, 0.8089, 0.9649, 0.8089)
Specificity <- c(0.99076, 0.90051, 0.7893, 0.7045, 0.3247, 0.7045)

metrics_df <- data.frame(models, Accuracy, Sensitivity, Specificity)
metrics_long <- pivot_longer(metrics_df, cols = c(Accuracy, Sensitivity, Specificity),
                             names_to = "metric_type", values_to = "value")


ggplot(metrics_long, aes(x = models, y = value, fill = metric_type, color = metric_type)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Performance Metrics by Model",
       x = "Model", y = "Value") +
  scale_color_manual(values = c("Accuracy" = "#6c8ca0", "Sensitivity" = "#8b668b", "Specificity" = "#dac1ac"),
                     name = "Metric") +
  scale_fill_manual(values = c("Accuracy" = "#6c8ca0", "Sensitivity" = "#8b668b", "Specificity" = "#dac1ac"),
                    name = "Metric") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



