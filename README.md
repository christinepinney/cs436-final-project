# cs436-final-project

To run this code, run cells to import necessary libraries. If any modules are not found, install the necessary packages. Any packages installed for this project that are not in the final_project notebooks can be found in `packages.ipynb`. 

### Notebook for Reddit dataset:
1. Run cell to install packages if necessary, then run cells to import necessary libraries.
2. Download dataset from Kaggle and put it in a dataframe.
3. Convert data to lowercase. Lemmatize and remove punctuation (or don't, to compare accuracies).
#### For Multinomial Naïve Bayes Classifier: 
4. Load data into a bag of words using `CountVectorizer()` with `stop_words` from the provided `glasgow_stop_words.txt`.
5. Split data into train and test sets, fit data to `MultinomialNB()` classifier and `accuracy_score`, `f1_score`, and `roc_auc_score`, as well as `classification_report`.
#### For ELECTRA:
4. Split data into train and test set and reshape data using `LabelBinarizer()`.
5. 
