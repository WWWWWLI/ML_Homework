from torchvision import datasets
import numpy as np
from sklearn import svm
import joblib

train_dataset = datasets.MNIST(root=r'D:\研一\机器学习及其应用\datasets\Homework1\data',
                               train=True,
                               download=True)

test_dataset = datasets.MNIST(root=r'D:\研一\机器学习及其应用\datasets\Homework1\data',
                              train=False,
                              download=True)


# print(train_dataset)  # Number of datapoints: 60000
# print(test_dataset)  # Number of datapoints: 10000


# print(train_dataset[1][0])  # <PIL.Image.Image image mode=L size=28x28 at 0x2828254C348>
# print(train_dataset[1][1])  # 0
# print(test_dataset[1][0])  # <PIL.Image.Image image mode=L size=28x28 at 0x27B47040DC8>
# print(test_dataset[1][1])  # 2

def img2numpy(image):
    img_arr = np.array(image)
    img_arr = img_arr.reshape(784)
    return img_arr


train_x = np.zeros(shape=(60000, 784))
train_y = np.zeros(shape=(60000,))

for i, train_sample in enumerate(train_dataset):
    train_x[i] = img2numpy(train_sample[0])
    train_y[i] = train_sample[1]
train_x = (train_x / 255.0) - 0.5

test_x = np.zeros(shape=(10000, 784))
test_y = np.zeros(shape=(10000,))

for i, test_sample in enumerate(test_dataset):
    test_x[i] = img2numpy(test_sample[0])
    test_y[i] = test_sample[1]

test_x = test_x / 255.0 - 0.5

clf = svm.SVC(gamma='auto')
clf.fit(train_x, train_y)
joblib.dump(clf, 'clf_norm.pkl')
# clf = joblib.load('clf.pkl')
print(clf.score(test_x, test_y))
