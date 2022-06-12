from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
   
@app.route('/predict', methods=['POST'])
def predict():

    #Loading the data from csv file to a pandas Dataframe
    raw_mail_data = pd.read_csv('mail_data.csv')
        
    #Replace the null values with a null string
    mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')
    
    #Label spam mail as 0;  ham mail as 1;
    mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1
    
    #Separating the data as texts and label
    #X-input
    #Y-Output/target
    X = mail_data['Message']
    Y = mail_data['Category']
    
    #Splitting the data into training data & test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    
    #Transform the text data to feature vectors that can be used as input to the Logistic regression
    feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

    #Splited X has string values, those need to be fit & converted to integer
    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)

    #Convert Y_train and Y_test values as integers [convert object type to int]
    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')
    
    #Training the Model
    model = LogisticRegression()

    #Training the Logistic Regression model with the training data
    model.fit(X_train_features, Y_train)

    #Prediction on training data
    prediction_on_training_data = model.predict(X_train_features)
    accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

    #Prediction on test data
    prediction_on_test_data = model.predict(X_test_features)
    accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

    if request.method=='POST':
        comment=request.form['comment']
        data=[comment]
            
        #Convert text to feature vectors
        input_data_features = feature_extraction.transform(data).toarray()
        
        #Making prediction
        my_prediction = model.predict(input_data_features)

    return render_template('result.html', prediction=my_prediction)

if __name__== '__main__':
    app.run(debug=True)
