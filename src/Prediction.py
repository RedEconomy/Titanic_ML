from sklearn.metrics import accuracy_score
prediction = model.predict(X_test)
accuracy_score(Y_test, prediction)
