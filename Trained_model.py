
# Saving the trained model for future prediction
import pickle

filename = 'trained_model.sav'
pickle.dump(model, open(filename,'wb'))

# Using the trained model to make new prediction
# Loading the saved model
loaded_model = pickle.load(open('/trained_model.sav', 'rb'))
X_new = X_test[2000]
print(Y_test[2000])

prediction = model.predict(X_new)
if (prediction[0] == -1):
  print('Negative sentiment')
elif (prediction[0] == 1):
  print('Positive sentiment')
else:
  print('Neutral sentiment')