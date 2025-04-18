# -----------------------------------
# Customer Churn Analysis with Logistic Regression
# -----------------------------------

# 0. Install missing packages (run only once)
packages <- c("tidyverse", "caret", "data.table", "ggplot2", "corrplot", "pROC", "reshape2")
installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

# 1. Load required libraries
library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)
library(corrplot)
library(pROC)
library(reshape2)

# 2. Load & Prepare the Data
url <- "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/refs/heads/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
data <- fread(url)

# Drop ID column and clean TotalCharges
data <- data %>% select(-customerID)
data$TotalCharges <- as.numeric(data$TotalCharges)
data$TotalCharges[is.na(data$TotalCharges)] <- median(data$TotalCharges, na.rm = TRUE)
data <- data %>% mutate_if(is.character, as.factor)

# 3. Exploratory Data Analysis (EDA)

# Churn distribution
churn_plot <- ggplot(data, aes(x = Churn)) +
  geom_bar(fill = "#FF6F61") +
  labs(title = "Churn Distribution", x = "Churn", y = "Count") +
  theme_minimal(base_size = 14)
ggsave("images/churn_distribution.png", churn_plot, width = 6, height = 4)

# Correlation matrix
numeric_data <- select_if(data, is.numeric)
png("images/correlation_matrix.png", width = 800, height = 800)
corrplot(cor(numeric_data), method = "color", type = "upper", tl.col = "black", number.cex = 0.7)
dev.off()

# Boxplot of TotalCharges by Churn
total_charges_plot <- ggplot(data, aes(x = Churn, y = TotalCharges, fill = Churn)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Total Charges by Churn Status") +
  theme_minimal(base_size = 14)
ggsave("images/total_charges_boxplot.png", total_charges_plot, width = 6, height = 4)

# 4. Train-Test Split
set.seed(123)
data$Churn <- as.factor(data$Churn)
split_index <- createDataPartition(data$Churn, p = 0.8, list = FALSE)
train_data <- data[split_index, ]
test_data <- data[-split_index, ]

# 5. Logistic Regression Model
model_log <- glm(Churn ~ ., data = train_data, family = "binomial")
summary(model_log)

# 6. Model Evaluation
pred_log <- predict(model_log, test_data, type = "response")
pred_class_log <- ifelse(pred_log > 0.5, "Yes", "No") %>% as.factor()

# Confusion Matrix
cm_log <- confusionMatrix(pred_class_log, test_data$Churn)
print(cm_log)

# Confusion Matrix Plot - Fixed Orientation
cm_data <- as.data.frame(cm_log$table)
cm_data$Reference <- factor(cm_data$Reference, levels = rev(levels(cm_data$Reference)))

cm_plot <- ggplot(cm_data, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white", alpha = 0.8) +
  geom_text(aes(label = Freq), size = 6, color = "white", fontface = "bold") +
  scale_fill_gradient(low = "#D3E5FF", high = "#1E40AF", name = "Count") +
  labs(title = "Confusion Matrix: Logistic Regression", 
       x = "Predicted Class", 
       y = "Actual Class",
       subtitle = paste("Accuracy:", round(cm_log$overall['Accuracy'], 3))) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right",
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        panel.grid = element_blank()) +
  coord_fixed()  # Ensures square tiles

ggsave("images/confusion_matrix_plot.png", cm_plot, width = 7, height = 5, dpi = 300)

# ROC & AUC
roc_log <- roc(test_data$Churn, pred_log)
auc_log <- auc(roc_log)

png("roc_curve.png", width = 600, height = 600)
plot(roc_log, col = "#E63946", lwd = 3, main = "ROC Curve - Logistic Regression")
legend("bottomright", legend = paste0("AUC = ", round(auc_log, 3)), col = "#E63946", lwd = 3)
dev.off()

# 7. Final AUC Output
cat("✅ AUC (Logistic Regression):", round(auc_log, 3), "\n")