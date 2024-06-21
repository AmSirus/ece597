# Task Distribution

- **Dingyang Miao**: Data preprocessing
- **Li Zhang**: Random Forest
- **Jiaxing Yao**: SVM
- I successfully completed the construction of the phishing email detection model and performed preliminary optimization and evaluation.
- First, because the dataset was extremely unbalanced, I adjusted the ratio of normal emails to phishing emails to 10:1.After that, I set class_weight to be balanced in the SVM parameter settings to further balance the ratio of normal emails to phishing emails.
- Then, text vectorization was performed by TF-IDF and the SVM model was used for training.
- Hyper-parameters were optimized by grid search and cross-validation, and hyper-parameter filtering was performed based on the recall's scores, which ultimately resulted in the best model parameters so far.
- The accuracy of the model on the training and test sets is 90.88% and 91.68%, respectively, and the precision, recall and F1 score on the test set are 52.43%, 91.89% and 66.76%, respectively.

- **Lian Duan**: Naive Bayes

- **Xiaotian Gan**: LLM - t5-base
Individual-Participation
06/20
This week, I focused on enhancing a text classification model to accurately distinguish between normal and phishing emails. I fine-tuned a T5 transformer model, troubleshooting and optimizing various parameters and data preprocessing steps. My efforts included debugging code errors, refining the model's predictions to output binary results, and ensuring the accuracy of predictions through meticulous testing and adjustments.

