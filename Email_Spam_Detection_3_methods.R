# Import necessary libraries

library(tm)
library(SnowballC)
library(e1071)
library(caret)
library(ggplot2)
library(randomForest)



##  DATASET (LOAD AND MODIFICATION)

# Read the spam collection dataset
data_text <- read.delim("SpamCollection", sep="\t", header=F, colClasses="character", quote="")
# Examine the data structure
str(data_text)
# Display the first few rows of the data
head(data_text)
# Extract the column names
colnames(data_text)
# Rename the columns
colnames(data_text) <- c("Class", "Text")
# Convert the "Class" column to a factor with two levels
data_text$Class <- factor(data_text$Class)
# Display the proportion of each class
prop.table(table(data_text$Class))



##  TEXT CLEANING 

# Clean the texts
corpus = VCorpus(VectorSource(data_text$Text))
# Convert all text to lowercase
corpus = tm_map(corpus, content_transformer(tolower))
# Remove numbers from the text
corpus = tm_map(corpus, removeNumbers)
# Remove punctuation from the text
corpus = tm_map(corpus, removePunctuation)
# Remove stop words from the text
corpus = tm_map(corpus, removeWords, stopwords("english"))
# Stem the words in the text
corpus = tm_map(corpus, stemDocument)
# Strip whitespace from the text
corpus = tm_map(corpus, stripWhitespace)



## BAG OF WORDS

# Create a document-term matrix
dtm = DocumentTermMatrix(corpus)
# Remove sparse terms from the document-term matrix
dtm = removeSparseTerms(dtm, 0.999)
# Display the dimensions of the document-term matrix
dim(dtm)
# Inspect a portion of the document-term matrix
inspect(dtm[40:50, 10:15])
# Define a function to convert counts to binary values
convert_count <- function(x) {
  ifelse(x > 0, 1, 0)
}
# Apply the convert_count function to get final training and testing DTMs
datasetNB <- apply(dtm, 2, convert_count)
# Convert the document-term matrix to a data frame
dataset = as.data.frame(as.matrix(datasetNB))



##  VISUALIZE FREQUENCY OF WORDS

# Identify frequently occurring terms
freq <- sort(colSums(as.matrix(dtm)), decreasing = TRUE)
tail(freq, 10)
# Find frequently appearing terms
findFreqTerms(dtm, lowfreq = 60)
# Plot word frequencies
wf <- data.frame(word = names(freq), freq = freq)
head(wf)
pp <- ggplot(subset(wf, freq > 100), aes(x = reorder(word, -freq), y = freq)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
# Display the plot
pp



##  SPLIT DATA

# Add the Class label back to the dataset
dataset$Class = data_text$Class
str(dataset$Class)
# Set the seed for reproducibility
set.seed(222)
# Split the data into training and testing sets
split <- sample(2, nrow(dataset), prob = c(0.75, 0.25), replace = TRUE)
train_set <- dataset[split == 1,]
test_set <- dataset[split == 2,]
# Display the proportion of each class in the training set
prop.table(table(train_set$Class))
# Display the proportion of each class in the testing set
prop.table(table(test_set$Class))



##  RANDOM FOREST

# Train a Random forest classifier
rf_classifier = randomForest(x = train_set[-1210], y = train_set$Class, ntree = 300)
# Display the Random forest classifier summary
rf_classifier
# Predicting the Test set results
rf_pred = predict(rf_classifier, newdata = test_set[-1210])
# Evaluate the classifier performance using a confusion matrix
confusionMatrix(table(rf_pred,test_set$Class))



##  NAIVE BAYES CLASSIFIER 

# Create control limit for NB model
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# Train a Naive Bayes classifier
system.time( classifier_nb <- naiveBayes(train_set, train_set$Class, laplace = 1, trControl = control,tuneLength = 7) )
# Make predictions on the testing set
nb_pred = predict(classifier_nb, type = 'class', newdata = test_set)
# Evaluate the classifier performance using a confusion matrix
confusionMatrix(nb_pred,test_set$Class)


## SVM

# Train a support vector machine classifier
svm_classifier <- svm(Class~., data = train_set)
# Display the classifier summary
svm_classifier
# Make predictions on the testing set
svm_pred <- predict(svm_classifier, test_set)
# Evaluate the classifier performance using a confusion matrix
confusionMatrix(svm_pred, test_set$Class)



##  NEW DATA FROM USER (PREDICTION USING NB)

# Get user input for the email to be tested
email_text <- readline(prompt = "Enter the email you want to test: ")
# Create a new data frame for the user input
new_test_set <- data.frame(Text = email_text)
# Clean the user input text
new_corpus <- VCorpus(VectorSource(new_test_set$Text))
new_corpus <- tm_map(new_corpus, content_transformer(tolower))
new_corpus <- tm_map(new_corpus, removeNumbers)
new_corpus <- tm_map(new_corpus, removePunctuation)
new_corpus <- tm_map(new_corpus, removeWords, stopwords("english"))
new_corpus <- tm_map(new_corpus, stemDocument)
new_corpus <- tm_map(new_corpus, stripWhitespace)
# Create a document-term matrix for the user input
new_dtm <- DocumentTermMatrix(new_corpus)
# Convert the new document-term matrix to a data frame
new_test_set <- as.data.frame(as.matrix(new_dtm))
# Make a prediction on the user input
nb_pred <- predict(classifier_nb, type = 'class', newdata = new_test_set)
# Determine and display the spam or not spam message
if (nb_pred == "ham") {
  print("This email is likely not spam.")
} else {
  print("This email is likely spam.")
}