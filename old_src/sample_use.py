from logistic_regression import MulticlassLogisticRegression

num_features = len(x_train[0])
num_classes = len(set(y_train))

model = MulticlassLogisticRegression(
    num_features=num_features,
    num_classes=num_classes,
    learning_rate=0.1
)

model.fit(x_train, y_train, num_epochs=20)

# Validation accuracy
y_pred_val = model.predict(x_val)
correct = sum(int(p == t) for p, t in zip(y_pred_val, y_val))
val_acc = correct / len(y_val)
print("Validation accuracy:", val_acc)
