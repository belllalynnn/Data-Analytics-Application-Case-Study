#######################################################################
###########################  CASE STUDY 2  ############################
#######################################################################

library(MASS) ; library(ggplot2) ; library(scales) ; library(tidyverse) ; library(corrplot) 
library(car) ; library(caret) ; library(readxl) ; library(e1071)

# Loading data
bbbc_train <- read_excel("/Users/BellaIsCute/Desktop/DA Application/BBBC-Train.xlsx")
bbbc_test <- read_excel("/Users/BellaIsCute/Desktop/DA Application/BBBC-Train.xlsx")


# Take a look at the data
names(bbbc_train)
str(bbbc_train)
summary(bbbc_train)
bbbc_train$Choice = as.factor(bbbc_train$Choice)
bbbc_test$Choice = as.factor(bbbc_test$Choice)
bbbc_train$Gender = as.factor(bbbc_train$Gender)
bbbc_test$Gender = as.factor(bbbc_test$Gender)

# Dropping "Observation"
bbbc_train = select(bbbc_train, -Observation)
bbbc_test = select(bbbc_test, -Observation)

##### BALANCING DATA 
## Resample with more balanced train data 
train_yes = bbbc_train %>% filter(Choice == "1")
train_no = bbbc_train %>% filter(Choice == "0")
set.seed(1)
sample_no_train = sample_n(train_no, nrow(train_yes), replace = TRUE) #matching the same row number
train_bal = rbind(train_yes,sample_no_train) #combining both so they will have a balance data


## Resample with more balanced test data 
test_yes = bbbc_test %>% filter(Choice == "1")
test_no = bbbc_test %>% filter(Choice == "0")
set.seed(1)
sample_no_test = sample_n(test_no, nrow(test_yes), replace = TRUE) #matching the same row number
test_bal = rbind(test_yes,sample_no_test) #combining both so they will have a balance data




             ###############  Linear regression  ###############
resultsLM = lm(as.numeric(Choice) ~ ., data = train_bal)
summary(resultsLM)

predLM = predict(resultsLM, test_bal)  # Removed # from beginning of line
print(predLM)# Prints the prediction of the linear model (confusion matrix gave error since the linear regression predicts continuous values)

# Plot linear regression
ggplot(test_bal, aes(x = seq_along(Choice), y = Choice, color = "Actual")) +
  geom_point() +
  geom_point(aes(y = predLM, color = "Predicted")) +
  labs(title = "Comparison of Actual and Predicted Values",
       x = "Data Point",
       y = "Value") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal()



             ############### Logistic regression ###############

log.all = glm(Choice ~ .,data = train_bal, family = binomial)
summary(log.all)
vif(log.all)

# Removing Last_purchase
log.all = glm(formula = Choice ~ . -Last_purchase, data = train_bal, family = binomial)
summary(log.all)
vif(log.all)

# Model
predprob_log <- predict.glm(log.all, test_bal, type = "response")
predclass_log = ifelse(predprob_log >= 0.5, yes = 1, 0)

# Confusion matrix
caret::confusionMatrix(as.factor(predclass_log), test_bal$Choice, positive = "1")
# Accuracy    : 0.7304 
# Sensitivity : 0.6667          
# Specificity : 0.7941  


######  Stepwise Selection with AIC
null_model = glm(Choice ~ 1, data = train_bal, family = binomial) 
full_model = log.all 

step.model.AIC = step(null_model, scope = list(upper = full_model),
                      direction = "both", test = "Chisq", trace = F) 
summary(step.model.AIC) 

# Best model based on stepwise 
log.sel <- glm(Choice ~ P_Art + Frequency + Gender + P_DIY + P_Cook + P_Child +
            First_purchase + Amount_purchased, train_bal, family = binomial)
summary(log.sel)

# Predict the responses on the testing data
test_bal$PredProb = predict.glm(log.sel, newdata = test_bal, type = "response") 
test_bal$PredY = ifelse(test_bal$PredProb >= 0.5, 1,0)

caret::confusionMatrix(as.factor(test_bal$Choice), as.factor(test_bal$PredY), positive = "1")
# Accuracy    : 0.7353
# Sensitivity : 0.7667          
# Specificity : 0.7105 



             ###############        SVM         ###############
set.seed(1) ; tuned = tune.svm(Choice ~ ., data = train_bal, gamma = seq(.01, .1, by = .01),
                                       cost = seq(1, 10, by = .1)) 

tuned$best.parameters # Gamma: 0.03  # Cost 2.6


## SVM - Kernel: RBF
SVM.rbf = svm(Choice ~ ., data = train_bal, gamma = tuned$best.parameters$gamma,
                                            cost = tuned$best.parameters$cost)
summary(SVM.rbf)

rbf_svm_predict = predict(SVM.rbf, test_bal, type = "response")
table(pred = rbf_svm_predict, true = test_bal$Choice)
caret::confusionMatrix(as.factor(rbf_svm_predict), as.factor(test_bal$Choice), positive = "1")
# Accuracy    : 0.7304
# Sensitivity : 0.6176          
# Specificity : 0.8431 


## SVM - Kernel: Linear
SVM.lin = svm(Choice ~ ., data = train_bal, gamma = tuned$best.parameters$gamma,
               cost = tuned$best.parameters$cost, kernel = "linear")

summary(SVM.lin)

linear_svm_predict = predict(SVM.lin, test_bal, type = "response")
table(pred = linear_svm_predict, true = test_bal$Choice)
caret::confusionMatrix(as.factor(linear_svm_predict), as.factor(test_bal$Choice), positive = "1")
# Accuracy    : 0.7304
# Sensitivity : 0.6275          
# Specificity : 0.8333  


## SVM - Kernel: Polynomial
SVM.poly = svm(Choice ~ ., data = train_bal, gamma = tuned$best.parameters$gamma,
               cost = tuned$best.parameters$cost, kernel = "polynomial")

summary(SVM.poly)

poly_svm_predict = predict(SVM.poly, test_bal, type = "response")
table(pred = poly_svm_predict, true = test_bal$Choice)
caret::confusionMatrix(as.factor(poly_svm_predict), as.factor(test_bal$Choice), positive = "1")
# Accuracy    : 0.6176 
# Sensitivity : 0.2451         
# Specificity : 0.9902 


## SVM - Kernel: Sigmoid 
SVM.sig = svm(Choice ~ ., data = train_bal, gamma = tuned$best.parameters$gamma,
               cost = tuned$best.parameters$cost, kernel = "sigmoid")

summary(SVM.sig)

sigmoid_svm_predict = predict(SVM.sig, test_bal, type = "response")
table(pred = sigmoid_svm_predict, true = test_bal$Choice)
caret::confusionMatrix(as.factor(sigmoid_svm_predict), as.factor(test_bal$Choice), positive = "1")
# Accuracy    : 0.7255
# Sensitivity : 0.6765          
# Specificity : 0.7745



         ################### Calculating Profit #####################

##### Sending to all:
cost_all = 50000 * 0.65
# 50,000 * Unit mailing cost

revenue_all = 0.0903 * 50000 * (31.95-15-6.75)
# Revenue: 50,000 * 9.03% * (Selling Price - Costs)

profit_all = (revenue_all - cost_all)
profit_all # 13,553



##### Using log.all to target customers:
cost_log.all = (42+136)/408 * 50000 * 0.65
#               (FP+TP)/408 * 50,000 * Unit mailing cost
#                      ^
#        Proportion of 1’s in model

revenue_log.all = 136/408 * 50000 * (31.95-15-6.75)
# Number of TP/Total test observations * 50,000 * (Selling Price - Book costs)

profit_log.all = (revenue_log.all - cost_log.all)
profit_log.all # $155,821.1


##### Using log.sel to target customers:
cost_log.sel = (66+138)/408 * 50000 * 0.65
#               (FP+TP)/408 * 50,000 * Unit mailing cost
#                      ^
#        Proportion of 1’s in model

revenue_log.sel = 138/408 * 50000 * (31.95-15-6.75)
# Number of TP/Total test observations * 50,000 * (Selling Price - Costs)

profit_log.sel = (revenue_log.sel - cost_log.sel)
profit_log.sel # $156,250


##### Using svm.rbf to target customers:
cost_svm.rbf = (32+126)/408 * 50000 * 0.65
#               (FP+TP)/408 * 50,000 * Unit mailing cost
#                      ^
#        Proportion of 1’s in model

revenue_svm.rbf = 126/408 * 50000 * (31.95-15-6.75)
# Number of TP/Total test observations * 50,000 * (Selling Price - Costs)

profit_svm.rbf = (revenue_svm.rbf - cost_svm.rbf)
profit_svm.rbf # $144,914.2


##### Using svm.lin to target customers:
cost_svm.lin = (34+128)/408 * 50000 * 0.65
#               (FP+TP)/408 * 50,000 * Unit mailing cost
#                      ^
#        Proportion of 1’s in model

revenue_svm.lin = 128/408 * 50000 * (31.95-15-6.75)
# Number of TP/Total test observations * 50,000 * (Selling Price - Costs)

profit_svm.lin = (revenue_svm.lin - cost_svm.lin)
profit_svm.lin # $147,095.6


##### Using svm.poly to target customers:
cost_svm.poly = (2+50)/408 * 50000 * 0.65
#               (FP+TP)/408 * 50,000 * Unit mailing cost
#                      ^
#        Proportion of 1’s in model

revenue_svm.poly = 50/408 * 50000 * (31.95-15-6.75)
# Number of TP/Total test observations * 50,000 * (Selling Price - Costs)

profit_svm.poly = (revenue_svm.poly - cost_svm.poly)
profit_svm.poly # $58,357.84


##### Using svm.sig to target customers:
cost_svm.sig = (46+138)/408 * 50000 * 0.65
#               (FP+TP)/408 * 50,000 * Unit mailing cost
#                      ^
#        Proportion of 1’s in model

revenue_svm.sig = 138/408 * 50000 * (31.95-15-6.75)
# Number of TP/Total test observations * 50,000 * (Selling Price - Costs)

profit_svm.sig = (revenue_svm.sig - cost_svm.sig)
profit_svm.sig # $157,843.1


          ############### Evaluating Models on Profit ###############

data <- data.frame(model = c("log.all", "log.sel", "SVM.rbf", "SVM.lin", "SVM.poly", "SVM.sig"),
                   profit = c(155821.1, 156250, 144914.2, 147095.6, 58357.84, 157843.1))

custom_colors <- c("log.all" = "#5B9F9A", "log.sel" = "#7DAFCA", "SVM.rbf" = "#E9909D", 
                   "SVM.lin" = "#AABAE4", "SVM.poly" = "#D2C3EE", "SVM.sig" = "#DF63A7")

ggplot(data, aes(x = model, y = profit, fill = model)) +
  geom_bar(stat = "identity", width = 0.5, fill = custom_colors) +
  geom_hline(yintercept = 13553, linetype = "dashed") +
  geom_text(aes(label = paste0("$", comma(round(profit, 2)))), vjust = -0.5, size = 3) +
  geom_text(aes(label = "Without a Model"), x = 5.5, y = 14000,
            vjust = -0.5, size = 3) +
  labs(title = "Profit Comparison of Models", x = "", y = "Profit ($)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust = 1),
        legend.position = "none")



############# Illustrating the effect of balancing data  #############

#### log.all.unb
log.all.unb = glm(Choice ~ .,data = bbbc_train, family = binomial)
summary(log.all.unb)
vif(log.all.unb)

# Removing Last_purchase
log.all.unb = glm(formula = Choice ~ . -Last_purchase, data = bbbc_train, family = binomial)
summary(log.all.unb)
vif(log.all.unb)

##### log.sel.unb
null_model_unb = glm(Choice ~ 1, data = bbbc_train, family = binomial) 
full_model_unb = log.all.unb 

step.model.AIC.unb = step(null_model_unb, scope = list(upper = full_model_unb),
                          direction = "both", test = "Chisq", trace = F) 
summary(step.model.AIC.unb) 

# Best model based on stepwise 
log.sel.unb <- glm(Choice ~ P_Art + Frequency + Gender + P_DIY + P_Cook + P_Child +
                     First_purchase + Amount_purchased + P_Youth, bbbc_train, family = binomial)
summary(log.sel)

# Predict the responses on the testing data
bbbc_test$PredProb = predict.glm(log.sel, newdata = bbbc_test, type = "response") 
bbbc_test$PredY = ifelse(bbbc_test$PredProb >= 0.5, 1,0)

caret::confusionMatrix(as.factor(bbbc_test$Choice), as.factor(bbbc_test$PredY), positive = "1")
# Accuracy    : 0.7478
# Sensitivity : 0.2117          
# Specificity : 0.9600


##### Using log.sel.unb to target customers:
cost_log.sel.unb = (66+138)/2300 * 50000 * 0.65
#                   (FP+TP)/2300 * 50,000 * Unit mailing cost
#                          ^
#            Proportion of 1’s in model

revenue_log.sel.unb = 138/2300 * 50000 * (31.95-15-6.75)
# Number of TP/Total test observations * 50,000 * (Selling Price - Costs)

profit_log.sel.unb = (revenue_log.sel.unb - cost_log.sel.unb)
profit_log.sel.unb # $27,717.39


###############   Data Diagnostics  ###############

par(mfrow = c(3, 3))
hist(train_bal$Amount_purchased, main = "Distribution of Amount_purchased", xlab = "")
hist(train_bal$Frequency, main = "Distribution of Frequency", xlab = "")
hist(train_bal$Last_purchase, main = "Distribution of Last_purchase", xlab = "")
hist(train_bal$First_purchase, main = "Distribution of First_purchase", xlab = "")
hist(train_bal$P_Child, main = "Distribution of P_Child", xlab = "")
hist(train_bal$P_Youth, main = "Distribution of P_Youth", xlab = "")
hist(train_bal$P_Cook, main = "Distribution of P_Cook", xlab = "")
hist(train_bal$P_DIY, main = "Distribution of P_DIY", xlab = "")
hist(train_bal$P_Art, main = "Distribution of P_Art", xlab = "")