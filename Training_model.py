# Training the model using logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train,Y_train)

# Model Evaluation 
# Accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print(training_data_accuracy)

