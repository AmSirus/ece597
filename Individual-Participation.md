# Task Distribution

- ## **Dingyang Miao**: Data preprocessing
    ### Comparing Two Datasets (Normal Emails and Abnormal Emails):
    Extract common features shared by both datasets.
    Remove the remaining parts that are not common between the two datasets.
    
    ### Cleaning the Data Content:
    Convert text to lowercase
    Remove HTML tags
    Remove URLs
    Remove email addresses
    Remove special characters and punctuation
    Remove repeated characters
    Tokenize the text
    Remove stopwords
    Perform lemmatization
    Normalize common abbreviations
    ### Extracting Data Features:
    Use TfidfVectorizer to extract the most frequently occurring words from both normal and abnormal emails based on term frequency.

  
- ## **Li Zhang**: Random Forest
- 6/20
- I completed preparing the data for model training, including splitting the data and resampling the training set. I also used both a dummy and random forest classifier, leveraging grid search and cross-validation to optimize the tuning parameters. To handle imbalanced data, I utilized the "class_weight" parameter within the Random Forest classifier and applied oversampling techniques from the imblearn library. In the following steps, I will try to implement embeddings using Word2Vec and the Bag-of-Words approach.
- 
- ## **Jiaxing Yao**: SVM
- I successfully completed the construction of the phishing email detection model and performed preliminary optimization and evaluation.
- First, because the dataset was extremely unbalanced, I adjusted the ratio of normal emails to phishing emails to 10:1.After that, I set class_weight to be balanced in the SVM parameter settings to further balance the ratio of normal emails to phishing emails.
- Then, text vectorization was performed by TF-IDF and the SVM model was used for training.
- Hyper-parameters were optimized by grid search and cross-validation, and hyper-parameter filtering was performed based on the recall's scores, which ultimately resulted in the best model parameters so far.
- The accuracy of the model on the training and test sets is 90.88% and 91.68%, respectively, and the precision, recall and F1 score on the test set are 52.43%, 91.89% and 66.76%, respectively.

- ## **Lian Duan**: Naive Bayes
 Plan to implement a Naive Bayes classifier to address the machine learning task. The process began with data preparation, including loading the dataset, handling missing values, and encoding categorical variables. I then split the data into training and testing sets and addressed class imbalance using the SMOTE technique. For feature extraction, I useTfidfVectorizer to convert text data into numerical features and train a MultinomialNB classifier and performed hyperparameter tuning using GridSearchCV to optimize parameters

- ## **Xiaotian Gan**: LLM - t5-base
Individual-Participation
- 06/20
- This week, I focused on enhancing a text classification model to accurately distinguish between normal and phishing emails. I fine-tuned a T5 transformer model, troubleshooting and optimizing various parameters and data preprocessing steps. My efforts included debugging code errors, refining the model's predictions to output binary results, and ensuring the accuracy of predictions through meticulous testing and adjustments.
- 6/28
- This week, we focused on refining and debugging a machine learning model using the T5 architecture in a Python environment. We addressed various challenges related to data preprocessing, model training, and the utilization of specific utilities such as Accelerate for optimized training. Issues such as input tensor dimensions, tokenizer warnings, and configuration mismatches were systematically resolved. Additionally, we worked on ensuring the model could correctly interpret and process input data to improve its training efficiency and output accuracy. Through iterative testing and adjustments, we aimed to enhance model performance and streamline the training process using advanced techniques like mixed precision training.

