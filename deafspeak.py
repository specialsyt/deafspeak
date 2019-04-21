import numpy as np
import cv2
import pandas
import time
import tensorflow as tf
import os


def set_variables():
    x_train = np.load('Training/x_train.npy')
    x_test = np.load('Training/x_test.npy')
    y_train = np.load('Training/y_train.npy')
    y_test = np.load('Training/y_test.npy')
    return x_train, x_test, y_train, y_test


def read_data(trainingDir, testingDir):
    dataset_train = pandas.read_csv(trainingDir)

    values_train = dataset_train.values

    xtrain = np.ndarray(shape=(len(values_train), 28, 28), dtype=np.uint8)
    ytrain = np.ndarray(shape=(len(values_train)), dtype=np.uint8)

    for value in range(len(values_train)):
        correct = values_train[value][0]
        image = np.delete(values_train[value], 0)
        for c in range(len(image)):
            x = c // 28
            y = c % 28
            xtrain[value][x][y] = image[c]
        ytrain[value] = correct

    dataset_test = pandas.read_csv(testingDir)

    values_test = dataset_test.values

    xtest = np.ndarray(shape=(len(values_test), 28, 28), dtype=np.uint8)
    ytest = np.ndarray(shape=(len(values_test)), dtype=np.uint8)

    for value in range(len(values_test)):
        correct = values_test[value][0]
        image = np.delete(values_test[value], 0)
        for c in range(len(image)):
            x = c // 28
            y = c % 28
            xtest[value][x][y] = image[c]
        ytest[value] = correct
    xtrain, xtest = xtrain / 255.0, xtest / 255.0

    np.save('Training/x_train', xtrain)
    np.save('Training/x_test', xtest)
    np.save('Training/y_train', ytrain)
    np.save('Training/y_test', ytest)

    return xtrain, xtest, ytrain, ytest


def create_model(layers):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(layers, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, epochs, checkpoint_path):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        # Save weights, every 5-epochs.
        period=10)
    model.save_weights(checkpoint_path.format(epoch=0))
    model.fit(x_train, y_train,
          epochs = epochs, callbacks = [cp_callback],
          validation_data=(x_test, y_test),
          verbose=0)
    return model


# x_train, x_test, y_train, y_test = read_data('sign-language-mnist/sign_mnist_train.csv', 'sign-language-mnist/sign_mnist_test.csv')
x_train, x_test, y_train, y_test = set_variables()
print('Loaded and Saved numPy data!')

# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "Model-Training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# model = train_model(model, 200, checkpoint_path)

latest = tf.train.latest_checkpoint(checkpoint_dir)

model = create_model(25)
model.load_weights(latest)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

predictions = model.predict(x_test)


# Write some Text

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 470)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2


for predict_index in range(len(predictions)):
    max_index = 0
    max_value = 0
    for x in range(len(predictions[predict_index])):
        if predictions[predict_index][x] > max_value:
            max_value = predictions[predict_index][x]
            max_index = x
    image = x_test[predict_index]
    resized_image = cv2.resize(image, (500, 500))
    prediction = str(max_index)
    correct = str(y_test[predict_index])
    cv2.putText(resized_image, 'Predicted: ' + prediction,
                (10, 440),
                font,
                fontScale,
                fontColor,
                lineType)
    cv2.putText(resized_image, 'Correct: ' + correct,
                (10, 470),
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.imshow('Sign Language', resized_image)
    time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
