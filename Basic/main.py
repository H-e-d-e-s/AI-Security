#tensorflow 2.0
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10, cifar100, fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D , UpSampling2D, Input

from collections import deque
import random

# sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
# numpy
import numpy as np
# matplotlib
import matplotlib.pyplot as plt
# art
from art.estimators.classification.tensorflow import TensorFlowV2Classifier
from art.attacks.evasion.boundary import BoundaryAttack
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.estimators.classification import KerasClassifier

import csv

tf.compat.v1.disable_eager_execution()
# load data and preprocess cifar10
(x_train_cif10, y_train_cif10), (x_test_cif10, y_test_cif10) = cifar10.load_data()
x_train_cif10 = x_train_cif10.astype("float32") / 255
x_test_cif10 = x_test_cif10.astype("float32") / 255
y_train_cif10 = to_categorical(y_train_cif10, num_classes=10)
y_test_cif10 = to_categorical(y_test_cif10, num_classes=10)

# load data and preprocess cifar100
(x_train_cif100, y_train_cif100), (x_test_cif100, y_test_cif100) = cifar100.load_data()
x_train_cif100 = x_train_cif100.astype("float32") / 255
x_test_cif100 = x_test_cif100.astype("float32") / 255
y_train_cif100 = to_categorical(y_train_cif100, num_classes=100)
y_test_cif100 = to_categorical(y_test_cif100, num_classes=100)

# load data and preprocess fashion_mnist
(x_train_fmnist, y_train_fmnist), (x_test_fmnist, y_test_fmnist) = fashion_mnist.load_data()
x_train_fmnist = x_train_fmnist.astype("float32") / 255
x_test_fmnist = x_test_fmnist.astype("float32") / 255
y_train_fmnist = to_categorical(y_train_fmnist, num_classes=10)
y_test_fmnist = to_categorical(y_test_fmnist, num_classes=10)


# Create a model using TensorFlow
print("Training model CIFAR10")
model_cif10 = Sequential([
    Conv2D(32, 3, activation="relu", input_shape=(32, 32, 3)),
    Conv2D(32, 3, activation="relu", input_shape=(32, 32, 3)),
    Conv2D(32, 3, activation="relu", input_shape=(32, 32, 3)),
    Flatten(),
    Dense(10, activation="softmax")
])

model_cif10.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model_cif10.fit(x_train_cif10, y_train_cif10, epochs=10, batch_size=64)
score = model_cif10.evaluate(x_test_cif10, y_test_cif10, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

classifier_10 = KerasClassifier(
    model=model_cif10,
    clip_values=(0,1),
    use_logits=False
)

model_cif100 = Sequential([
    Conv2D(64, 3, activation="relu", input_shape=(32, 32, 3)),
    Conv2D(64, 3, activation="relu"),
    Flatten(),
    Dense(256, activation="relu"),
    Dense(100, activation="softmax")
])

print("Training model CIFAR100")
model_cif100.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model_cif100.fit(x_train_cif100, y_train_cif100, epochs=10, batch_size=64)
score = model_cif100.evaluate(x_test_cif100, y_test_cif100, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model_fmnist = Sequential([
    Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
    Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
    Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
    Flatten(),
    Dense(10, activation="softmax")
])
print("Training model Fashion MNIST")
model_fmnist.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

x_train_fmnist = x_train_fmnist.reshape(x_train_fmnist.shape[0], x_train_fmnist.shape[1], x_train_fmnist.shape[2], 1)
x_test_fmnist = x_test_fmnist.reshape(x_test_fmnist.shape[0], x_test_fmnist.shape[1], x_test_fmnist.shape[2], 1)
model_fmnist.fit(x_train_fmnist, y_train_fmnist, epochs=10, batch_size=64)
score = model_fmnist.evaluate(x_test_fmnist, y_test_fmnist, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#classifier = TensorFlowV2Classifier(model=model, nb_classes=10, input_shape=(32, 32, 3), clip_values=(0, 1))

#Attack the model using Fast Gradient Sign Method and Boundary Attack
attack_FGSM_10 = FastGradientMethod(
    estimator=classifier_10, 
    eps=0.3
   )
attack_Boundary_10 = BoundaryAttack(classifier_10)

classifier_100 = KerasClassifier(
    model=model_cif100,
    clip_values=(0,1),
    use_logits=False
)

classifier_fmnist = KerasClassifier(
    model=model_fmnist,
    clip_values=(0,1),
    use_logits=False
)
attack_FGSM_100 = FastGradientMethod(
    estimator=classifier_100,
    eps=0.3
    )
attack_Boundary_100 = BoundaryAttack(classifier_100)

attack_FGSM_fmnist = FastGradientMethod(
    estimator=classifier_fmnist,
    eps=0.3
    )
attack_Boundary_fmnist = BoundaryAttack(classifier_fmnist)




# Generate adversarial samples FGSM cifar10
print("Generating adversarial samples for CIFAR10")
x_adv_FGSM_cif10 = []
for i in range(10):
    x_FGSM_10 = x_test_cif10[i][np.newaxis, :]  # add an additional dimension to the input data
    y_FGSM_10= y_test_cif10[i]
    x_adv_FGSM_cif10.append(attack_FGSM_10.generate(x=x_FGSM_10, y=y_FGSM_10))
x_adv_FGSM_cif10= np.concatenate(x_adv_FGSM_cif10, axis=0)
# Generate adversarial samples Boundary cifar10
x_adv_BDRY_cif10 = []
for i in range(10):
    x_BDRY_10 = x_test_cif10[i][np.newaxis, :]  # add an additional dimension to the input data
    y_BDRY_10 = y_test_cif10[i]
    x_adv_BDRY_cif10.append(attack_Boundary_10.generate(x=x_BDRY_10, y=y_BDRY_10))
x_adv_BDRY_cif10 = np.concatenate(x_adv_BDRY_cif10, axis=0)

# Generate adversarial samples FGSM cifar100
print("Generating adversarial samples for CIFAR100")
x_adv_FGSM_cif100 = []
for i in range(10):
    x_FGSM_100 = x_test_cif100[i][np.newaxis, :]  # add an additional dimension to the input data
    y_FGSM_100= y_test_cif100[i]
    x_adv_FGSM_cif100.append(attack_FGSM_100.generate(x=x_FGSM_100, y=y_FGSM_100))
x_adv_FGSM_cif100 = np.concatenate(x_adv_FGSM_cif100, axis=0)
# Generate adversarial samples Boundary cifar100
x_adv_BDRY_cif100 = []
for i in range(10):
    x_BDRY_100 = x_test_cif100[i][np.newaxis, :]  # add an additional dimension to the input data
    y_BDRY_100 = y_test_cif100[i]
    x_adv_BDRY_cif100.append(attack_Boundary_100.generate(x=x_BDRY_100, y=y_BDRY_100))
x_adv_BDRY_cif100 = np.concatenate(x_adv_BDRY_cif100, axis=0)

# Generate adversarial samples FGSM fmnist
print("Generating adversarial samples for Fashion MNIST")
x_adv_FGSM_fmnist = []
for i in range(10):
    x_FGSM_fmnist = x_test_fmnist[i][np.newaxis, :]  # add an additional dimension to the input data
    y_FGSM_fmnist= y_test_fmnist[i]
    x_adv_FGSM_fmnist.append(attack_FGSM_fmnist.generate(x=x_FGSM_fmnist, y=y_FGSM_fmnist))
x_adv_FGSM_fmnist = np.concatenate(x_adv_FGSM_fmnist, axis=0)
# Generate adversarial samples Boundary fmnist
x_adv_BDRY_fmnist = []
for i in range(10):
    x_BDRY_fmnist = x_test_fmnist[i][np.newaxis, :]  # add an additional dimension to the input data
    y_BDRY_fmnist = y_test_fmnist[i]
    x_adv_BDRY_fmnist.append(attack_Boundary_fmnist.generate(x=x_BDRY_fmnist, y=y_BDRY_fmnist))
x_adv_BDRY_fmnist = np.concatenate(x_adv_BDRY_fmnist, axis=0)


# plot 10 images and their adversarial counter parts using FGSM : cifar10
fig, axes = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    axes[0, i].imshow(x_test_cif10[i])
    axes[1, i].imshow(x_adv_FGSM_cif10[i])
plt.title("FGSM")
fig.savefig('10imgFGSM.png')
plt.show()

# plot the difference between the original and adversarial images : cifar10
fig, axes = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    axes[i].imshow(x_adv_FGSM_cif10[i] - x_test_cif10[i])
plt.title("FGSM Difference")
fig.savefig('10imgFGSMDiff.png')
plt.show()

# plot 10 images and their adversarial counter parts using Boundary : cifar10
fig, axes = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    axes[0, i].imshow(x_test_cif10[i])
    axes[1, i].imshow(x_adv_BDRY_cif10[i])
plt.title("Boundary Attack")
fig.savefig('10imgBDRY.png')
plt.show()

# plot the difference between the original and adversarial images : cifar10
fig, axes = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    axes[i].imshow(x_adv_BDRY_cif10[i] - x_test_cif10[i])
plt.title("Boundary Difference")
fig.savefig('10imgBDRYDiff.png')
plt.show()

# plot the image and its label side by side with the adversarial image and its label using FGSM with the name of the class : cifar10
fig, axes = plt.subplots(2, 10, figsize=(10, 2))
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    axes[0, i].imshow(x_test_cif10[i])
    axes[1, i].imshow(x_adv_FGSM_cif10[i])
    axes[0, i].set_title(class_names[np.argmax(y_test_cif10[i])])
    axes[1, i].set_title(class_names[np.argmax(classifier_10.predict(x_adv_FGSM_cif10[i][np.newaxis, :]))])
fig.suptitle('FGSM', fontsize=16)
fig.savefig('10imgFGSMClass.png')
plt.show()

# plot the image and its label side by side with the adversarial image and its label using Boundary with the name of the class : cifar10
fig, axes = plt.subplots(2, 10, figsize=(10, 2))
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    axes[0, i].imshow(x_test_cif10[i])
    axes[1, i].imshow(x_adv_BDRY_cif10[i])
    axes[0, i].set_title(class_names[np.argmax(y_test_cif10[i])])
    axes[1, i].set_title(class_names[np.argmax(classifier_10.predict(x_adv_BDRY_cif10[i][np.newaxis, :]))])
fig.suptitle('Boundary Attack', fontsize=16)
fig.savefig('10imgBDRYClass.png')
plt.show()

# Evaluate the classifier on the adversarial examples using FGSM : cifar10
predictions_FGSM_cif10 = classifier_10.predict(x_adv_FGSM_cif10)
accuracy = np.sum(np.argmax(predictions_FGSM_cif10, axis=1) == np.argmax(y_test_cif10[:10], axis=1)) / 10
print("Accuracy on adversarial samples FGSM : CIFAR10  {}%".format(accuracy * 100))

predictions_BDRY_cif10 = classifier_10.predict(x_adv_BDRY_cif10)
accuracy = np.sum(np.argmax(predictions_BDRY_cif10, axis=1) == np.argmax(y_test_cif10[:10], axis=1)) / 10
print("Accuracy on adversarial samples Boundary: CIFAR10 {}%".format(accuracy * 100))

# plot the evaluation of the classifier on the adversarial examples using FGSM : cifar10
fig, axes = plt.subplots(1, 2, figsize=(10, 2))
axes[0].bar(class_names, predictions_FGSM_cif10[0])
axes[1].bar(class_names, predictions_BDRY_cif10[0])
fig.suptitle('Evaluation of the classifier on the adversarial examples', fontsize=16)
fig.savefig('10imgEval.png')
plt.show()

# plot 10 images and their adversarial counter parts using FGSM : cifar100
fig, axes = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    axes[0, i].imshow(x_test_cif100[i])
    axes[1, i].imshow(x_adv_FGSM_cif100[i])
plt.title("FGSM Attack")
fig.savefig('10imgFGSM100.png')
plt.show()

# plot the difference between the original and adversarial images : cifar100
fig, axes = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    axes[i].imshow(x_adv_FGSM_cif100[i] - x_test_cif100[i])
plt.title("FGSM Difference")
fig.savefig('10imgFGSMDiff100.png')
plt.show()

# plot the image and its label side by side with the adversarial image and its label using FGSM with the name of the class : cifar100
fig, axes = plt.subplots(2, 10, figsize=(10, 2))
cifar100_labels = [
'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
'worm'
]
for i in range(10):
    axes[0, i].imshow(x_test_cif100[i])
    axes[1, i].imshow(x_adv_FGSM_cif100[i])
    axes[0, i].set_title(cifar100_labels[np.argmax(y_test_cif100[i])])
    axes[1, i].set_title(cifar100_labels[np.argmax(classifier_100.predict(x_adv_FGSM_cif100[i][np.newaxis, :]))])
fig.suptitle('FGSM', fontsize=16)
fig.savefig('10imgFGSMClass100.png')
plt.show()

# plot the image and its label side by side with the adversarial image and its label using Boundary with the name of the class : cifar100
fig, axes = plt.subplots(2, 10, figsize=(10, 2))
cifar100_labels = [
'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
'worm'
]
for i in range(10):
    axes[0, i].imshow(x_test_cif100[i])
    axes[1, i].imshow(x_adv_BDRY_cif100[i])
    axes[0, i].set_title(cifar100_labels[np.argmax(y_test_cif100[i])])
    axes[1, i].set_title(cifar100_labels[np.argmax(classifier_100.predict(x_adv_BDRY_cif100[i][np.newaxis, :]))])
fig.suptitle('Boundary Attack', fontsize=16)
fig.savefig('10imgBDRYClass100.png')
plt.show()

# Evaluate the classifier on the adversarial examples using FGSM : cifar100
predictions_FGSM_cif100 = classifier_100.predict(x_adv_FGSM_cif100)
accuracy = np.sum(np.argmax(predictions_FGSM_cif100, axis=1) == np.argmax(y_test_cif100[:10], axis=1)) / 10
print("Accuracy on adversarial samples FGSM: CIFAR100 {}%".format(accuracy * 100))
predictions_BDRY_cif100 = classifier_100.predict(x_adv_BDRY_cif100)
accuracy = np.sum(np.argmax(predictions_BDRY_cif100, axis=1) == np.argmax(y_test_cif100[:10], axis=1)) / 10
print("Accuracy on adversarial samples Boundary: CIFAR100 {}%".format(accuracy * 100))

# plot the evaluation of the classifier on the adversarial examples using FGSM : cifar100
fig, axes = plt.subplots(1, 2, figsize=(10, 2))
axes[0].bar(cifar100_labels, predictions_FGSM_cif100[0])
axes[1].bar(cifar100_labels, predictions_BDRY_cif100[0])
fig.suptitle('Evaluation of the classifier on the adversarial examples', fontsize=16)
fig.savefig('10imgEval100.png')
plt.show()

# plot 10 images and their adversarial counter parts using FGSM : fashion mnist
fig, axes = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    axes[0, i].imshow(x_test_fmnist[i])
    axes[1, i].imshow(x_adv_FGSM_fmnist[i])
plt.title("FGSM Attack")
fig.savefig('10imgFGSMfmnist.png')
plt.show()

# plot the difference between the original and adversarial images : fashion mnist
fig, axes = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    axes[i].imshow(x_adv_FGSM_fmnist[i] - x_test_fmnist[i])
plt.title("FGSM Difference")
fig.savefig('10imgFGSMDifffmnist.png')
plt.show()

# plot the image and its label side by side with the adversarial image and its label using FGSM with the name of the class : fashion mnist
fig, axes = plt.subplots(2, 10, figsize=(10, 2))
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
for i in range(10):
    axes[0, i].imshow(x_test_fmnist[i])
    axes[1, i].imshow(x_adv_FGSM_fmnist[i])
    axes[0, i].set_title(class_names[np.argmax(y_test_fmnist[i])])
    axes[1, i].set_title(class_names[np.argmax(classifier_fmnist.predict(x_adv_FGSM_fmnist[i][np.newaxis, :]))])
fig.suptitle('FGSM', fontsize=16)
fig.savefig('10imgFGSMClassfmnist.png')
plt.show()

# plot the image and its label side by side with the adversarial image and its label using Boundary with the name of the class : fashion mnist
fig, axes = plt.subplots(2, 10, figsize=(10, 2))
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
for i in range(10):
    axes[0, i].imshow(x_test_fmnist[i])
    axes[1, i].imshow(x_adv_BDRY_fmnist[i])
    axes[0, i].set_title(class_names[np.argmax(y_test_fmnist[i])])
    axes[1, i].set_title(class_names[np.argmax(classifier_fmnist.predict(x_adv_BDRY_fmnist[i][np.newaxis, :]))])
fig.suptitle('Boundary Attack', fontsize=16)
fig.savefig('10imgBDRYClassfmnist.png')
plt.show()

# Evaluate the classifier on the adversarial examples using FGSM : fashion mnist
predictions_FGSM_fmnist = classifier_fmnist.predict(x_adv_FGSM_fmnist)
accuracy = np.sum(np.argmax(predictions_FGSM_fmnist, axis=1) == np.argmax(y_test_fmnist[:10], axis=1)) / 10
print("Accuracy on adversarial samples FGSM: fashion mnist {}%".format(accuracy * 100))
predictions_BDRY_fmnist = classifier_fmnist.predict(x_adv_BDRY_fmnist)
accuracy = np.sum(np.argmax(predictions_BDRY_fmnist, axis=1) == np.argmax(y_test_fmnist[:10], axis=1)) / 10
print("Accuracy on adversarial samples Boundary: fashion mnist {}%".format(accuracy * 100))

# plot the evaluation of the classifier on the adversarial examples using FGSM : fashion mnist
fig, axes = plt.subplots(1, 2, figsize=(10, 2))
axes[0].bar(class_names, predictions_FGSM_fmnist[0])
axes[1].bar(class_names, predictions_BDRY_fmnist[0])
fig.suptitle('Evaluation of the classifier on the adversarial examples', fontsize=16)
fig.savefig('10imgEvalfmnist.png')
plt.show()

# Attack success rate CIFAR10
print("Attack success rate for FGSM attack CIFAR10: ", np.sum(np.argmax(predictions_FGSM_cif10, axis=1) != np.argmax(y_test_cif10[:10], axis=1)) / 10)
print("Attack success rate for Boundary attack CIFAR10: ", np.sum(np.argmax(predictions_BDRY_cif10, axis=1) != np.argmax(y_test_cif10[:10], axis=1)) / 10)

# Attack success rate CIFAR100
print("Attack success rate for FGSM attack CIFAR100: ", np.sum(np.argmax(predictions_FGSM_cif100, axis=1) != np.argmax(y_test_cif100[:10], axis=1)) / 10)
print("Attack success rate for Boundary attack CIFAR100: ", np.sum(np.argmax(predictions_BDRY_cif100, axis=1) != np.argmax(y_test_cif100[:10], axis=1)) / 10)

# Attack success rate fashion mnist
print("Attack success rate for FGSM attack fashion mnist: ", np.sum(np.argmax(predictions_FGSM_fmnist, axis=1) != np.argmax(y_test_fmnist[:10], axis=1)) / 10)
print("Attack success rate for Boundary attack fashion mnist: ", np.sum(np.argmax(predictions_BDRY_fmnist, axis=1) != np.argmax(y_test_fmnist[:10], axis=1)) / 10)


# F1 score for FGSM attack and Boundary attack CIFAR10
print("F1 score for FGSM attack CIFAR10: ", f1_score(np.argmax(y_test_cif10[:10], axis=1), np.argmax(predictions_FGSM_cif10, axis=1), average='macro'))
print("F1 score for Boundary attack CIFAR10: ", f1_score(np.argmax(y_test_cif10[:10], axis=1), np.argmax(predictions_BDRY_cif10, axis=1), average='macro'))

# F1 score for FGSM attack and Boundary attack CIFAR100
print("F1 score for FGSM attack CIFAR100: ", f1_score(np.argmax(y_test_cif100[:10], axis=1), np.argmax(predictions_FGSM_cif100, axis=1), average='macro'))
print("F1 score for Boundary attack CIFAR100: ", f1_score(np.argmax(y_test_cif100[:10], axis=1), np.argmax(predictions_BDRY_cif100, axis=1), average='macro'))

# F1 score for FGSM attack and Boundary attack fashion mnist
print("F1 score for FGSM attack fashion mnist: ", f1_score(np.argmax(y_test_fmnist[:10], axis=1), np.argmax(predictions_FGSM_fmnist, axis=1), average='macro'))
print("F1 score for Boundary attack fashion mnist: ", f1_score(np.argmax(y_test_fmnist[:10], axis=1), np.argmax(predictions_BDRY_fmnist, axis=1), average='macro'))

#accuracy of the FGSM attack and Boundary attack CIFAR10
print("Accuracy of the FGSM attack CIFAR10: ", accuracy_score(np.argmax(y_test_cif10[:10], axis=1), np.argmax(predictions_FGSM_cif10, axis=1)))
print("Accuracy of the Boundary attack CIFAR10: ", accuracy_score(np.argmax(y_test_cif10[:10], axis=1), np.argmax(predictions_BDRY_cif10, axis=1)))

#accuracy of the FGSM attack and Boundary attack CIFAR100
print("Accuracy of the FGSM attack CIFAR100: ", accuracy_score(np.argmax(y_test_cif100[:10], axis=1), np.argmax(predictions_FGSM_cif100, axis=1)))
print("Accuracy of the Boundary attack CIFAR100: ", accuracy_score(np.argmax(y_test_cif100[:10], axis=1), np.argmax(predictions_BDRY_cif100, axis=1)))

#accuracy of the FGSM attack and Boundary attack fashion mnist
print("Accuracy of the FGSM attack fashion mnist: ", accuracy_score(np.argmax(y_test_fmnist[:10], axis=1), np.argmax(predictions_FGSM_fmnist, axis=1)))
print("Accuracy of the Boundary attack fashion mnist: ", accuracy_score(np.argmax(y_test_fmnist[:10], axis=1), np.argmax(predictions_BDRY_fmnist, axis=1)))

#precision of the FGSM attack and Boundary attack CIFAR10
print("Precision of the FGSM attack CIFAR10: ", precision_score(np.argmax(y_test_cif10[:10], axis=1), np.argmax(predictions_FGSM_cif10, axis=1), average='macro'))
print("Precision of the Boundary attack CIFAR10: ", precision_score(np.argmax(y_test_cif10[:10], axis=1), np.argmax(predictions_BDRY_cif10, axis=1), average='macro'))

#precision of the FGSM attack and Boundary attack CIFAR100
print("Precision of the FGSM attack CIFAR100: ", precision_score(np.argmax(y_test_cif100[:10], axis=1), np.argmax(predictions_FGSM_cif100, axis=1), average='macro'))
print("Precision of the Boundary attack CIFAR100: ", precision_score(np.argmax(y_test_cif100[:10], axis=1), np.argmax(predictions_BDRY_cif100, axis=1), average='macro'))

#precision of the FGSM attack and Boundary attack fashion mnist
print("Precision of the FGSM attack fashion mnist: ", precision_score(np.argmax(y_test_fmnist[:10], axis=1), np.argmax(predictions_FGSM_fmnist, axis=1), average='macro'))
print("Precision of the Boundary attack fashion mnist: ", precision_score(np.argmax(y_test_fmnist[:10], axis=1), np.argmax(predictions_BDRY_fmnist, axis=1), average='macro'))

#recall of the FGSM attack and Boundary attack CIFAR10
print("Recall of the FGSM attack CIFAR10: ", recall_score(np.argmax(y_test_cif10[:10], axis=1), np.argmax(predictions_FGSM_cif10, axis=1), average='macro'))
print("Recall of the Boundary attack CIFAR10: ", recall_score(np.argmax(y_test_cif10[:10], axis=1), np.argmax(predictions_BDRY_cif10, axis=1), average='macro'))

#recall of the FGSM attack and Boundary attack CIFAR100
print("Recall of the FGSM attack CIFAR100: ", recall_score(np.argmax(y_test_cif100[:10], axis=1), np.argmax(predictions_FGSM_cif100, axis=1), average='macro'))
print("Recall of the Boundary attack CIFAR100: ", recall_score(np.argmax(y_test_cif100[:10], axis=1), np.argmax(predictions_BDRY_cif100, axis=1), average='macro'))

#recall of the FGSM attack and Boundary attack fashion mnist
print("Recall of the FGSM attack fashion mnist: ", recall_score(np.argmax(y_test_fmnist[:10], axis=1), np.argmax(predictions_FGSM_fmnist, axis=1), average='macro'))
print("Recall of the Boundary attack fashion mnist: ", recall_score(np.argmax(y_test_fmnist[:10], axis=1), np.argmax(predictions_BDRY_fmnist, axis=1), average='macro'))

#confusion matrix of the FGSM attack and Boundary attack CIFAR10
print("Confusion matrix of the FGSM attack CIFAR10: ", confusion_matrix(np.argmax(y_test_cif10[:10], axis=1), np.argmax(predictions_FGSM_cif10, axis=1)))
print("Confusion matrix of the Boundary attack CIFAR10: ", confusion_matrix(np.argmax(y_test_cif10[:10], axis=1), np.argmax(predictions_BDRY_cif10, axis=1)))

#confusion matrix of the FGSM attack and Boundary attack CIFAR100
print("Confusion matrix of the FGSM attack CIFAR100: ", confusion_matrix(np.argmax(y_test_cif100[:10], axis=1), np.argmax(predictions_FGSM_cif100, axis=1)))
print("Confusion matrix of the Boundary attack CIFAR100: ", confusion_matrix(np.argmax(y_test_cif100[:10], axis=1), np.argmax(predictions_BDRY_cif100, axis=1)))

#confusion matrix of the FGSM attack and Boundary attack fashion mnist
print("Confusion matrix of the FGSM attack fashion mnist: ", confusion_matrix(np.argmax(y_test_fmnist[:10], axis=1), np.argmax(predictions_FGSM_fmnist, axis=1)))
print("Confusion matrix of the Boundary attack fashion mnist: ", confusion_matrix(np.argmax(y_test_fmnist[:10], axis=1), np.argmax(predictions_BDRY_fmnist, axis=1)))




# Define the DQN architecture
def create_dqn(input_shape, num_actions):
   inputs = Input(shape=input_shape)
   layer = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
   layer = Conv2D(64, kernel_size=(3, 3), activation='relu')(layer)
   layer = Flatten()(layer)
   layer = Dense(64, activation='relu')(layer)
   outputs = Dense(num_actions, activation='linear')(layer)
   model = Model(inputs=inputs, outputs=outputs)
   model.compile(optimizer='adam', loss='mse')
   return model

# Define the environment
class AdversarialDetectionEnv:
   def __init__(self, classifier, x_clean, x_adv, y_true):
       self.classifier = classifier
       self.x_clean = x_clean
       self.x_adv = x_adv
       self.y_true = y_true
       self.action_space = 2 # 0 for clean, 1 for adversarial
       self.state = None
       self.reset()

   def reset(self):
       # Randomly select a clean or adversarial example as the initial state
       is_adv = random.choice([0, 1])
       idx = random.randint(0, len(self.x_clean) - 1)
       self.state = self.x_adv[idx] if is_adv else self.x_clean[idx]
       return self.state

   def step(self, action):
       # Use the classifier to predict the label of the current state
       pred = self.classifier.predict(self.state[np.newaxis, :])
       label = np.argmax(pred, axis=1)
       true_label = np.argmax(self.y_true, axis=1)
       # Check if the classifier's prediction is correct
       is_correct = label == true_label
       # Define the reward based on the action and correctness of the prediction
       if action == 0 and is_correct: # Predicted as clean and is correct
           reward = 1
       elif action == 1 and not is_correct: # Predicted as adversarial and is correct
           reward = 1
       else:
           reward = -1
       # Get the next state
       next_state = self.reset()
       done = False # For simplicity, we don't define an episode end condition
       return next_state, reward, done, {}

# Instantiate the environments
env_cif10 = AdversarialDetectionEnv(classifier_10, x_train_cif10, x_adv_FGSM_cif10, y_train_cif10)
env_cif100 = AdversarialDetectionEnv(classifier_100, x_train_cif100, x_adv_FGSM_cif100, y_train_cif100)
env_fmnist = AdversarialDetectionEnv(classifier_fmnist, x_train_fmnist, x_adv_FGSM_fmnist, y_train_fmnist)

# Instantiate the DQNs
input_shape = env_cif10.state.shape
num_actions = env_cif10.action_space
dqn_cif10 = create_dqn(input_shape, num_actions)
dqn_cif100 = create_dqn(input_shape, num_actions)
dqn_fmnist = create_dqn(input_shape, num_actions)

# Train the DQNs
memory = deque(maxlen=2000)
epsilon = 1.0 # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32

for episode in range(1000):
   state = env_cif10.reset()
   state = np.expand_dims(state, axis=0)
   done = False
   while not done:
       # Epsilon-greedy action selection
       if np.random.rand() <= epsilon:
           action = random.randrange(num_actions)
       else:
           action_values = dqn_cif10.predict(state)
           action = np.argmax(action_values[0])

       next_state, reward, done, _ = env_cif10.step(action)
       next_state = np.expand_dims(next_state, axis=0)

       # Store the experience in memory
       memory.append((state, action, reward, next_state, done))

       state = next_state

       # Experience replay
       if len(memory) > batch_size:
           minibatch = random.sample(memory, batch_size)
           for state, action, reward, next_state, done in minibatch:
               target = reward
               if not done:
                  target = reward + 0.95 * np.amax(dqn_cif10.predict(next_state)[0])
               target_f = dqn_cif10.predict(state)
               target_f[0][action] = target
               dqn_cif10.fit(state, target_f, epochs=1, verbose=0)

   # Update epsilon
   if epsilon > epsilon_min:
       epsilon *= epsilon_decay

# Define the autoencoder architecture
def create_autoencoder(input_shape):
   inputs = Input(shape=input_shape)
   # Encoder
   encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
   encoded = MaxPooling2D((2, 2), padding='same')(encoded)
   # Decoder
   decoded = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
   decoded = UpSampling2D((2, 2))(decoded)
   decoded = Conv2D(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(decoded)
   # Autoencoder
   autoencoder = Model(inputs, decoded)
   autoencoder.compile(optimizer='adam', loss='mse')
   return autoencoder

# Instantiate the autoencoders
input_shape_cif10 = (32, 32, 3) # Example input shape for CIFAR-10
autoencoder_cif10 = create_autoencoder(input_shape_cif10)

input_shape_cif100 = (32, 32, 3) # Example input shape for CIFAR-100
autoencoder_cif100 = create_autoencoder(input_shape_cif100)

input_shape_fmnist = (28, 28, 1) # Example input shape for Fashion-MNIST
autoencoder_fmnist = create_autoencoder(input_shape_fmnist)

# Train the autoencoders on clean data
autoencoder_cif10.fit(x_train_cif10, x_train_cif10, epochs=50, batch_size=256, shuffle=True)
autoencoder_cif100.fit(x_train_cif100, x_train_cif100, epochs=50, batch_size=256, shuffle=True)
autoencoder_fmnist.fit(x_train_fmnist, x_train_fmnist, epochs=50, batch_size=256, shuffle=True)

# Use the trained DQNs to detect adversarial inputs and the autoencoders to restore them
def restore_adversarial_inputs(dqn, autoencoder, x_adv, threshold=0.5):
   restored_images = []
   for x in x_adv:
       x = np.expand_dims(x, axis=0)
       action_values = dqn.predict(x)
       action = np.argmax(action_values[0])
       if action == 1: # The DQN predicts this is an adversarial input
           restored = autoencoder.predict(x)
           restored_images.append(restored.squeeze()) # Remove batch dimension
       else:
           restored_images.append(x.squeeze())
   return np.array(restored_images)

x_adv_detected_cif10 = restore_adversarial_inputs(dqn_cif10, autoencoder_cif10, x_adv_FGSM_cif10)
x_adv_detected_cif100 = restore_adversarial_inputs(dqn_cif100, autoencoder_cif100, x_adv_FGSM_cif100)
x_adv_detected_fmnist = restore_adversarial_inputs(dqn_fmnist, autoencoder_fmnist, x_adv_FGSM_fmnist)
