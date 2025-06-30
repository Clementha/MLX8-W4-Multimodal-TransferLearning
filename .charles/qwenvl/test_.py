import sys
import os
import matplotlib.pyplot as plt

# Fix the path to correctly point to the mnist_reader module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../.data/fashion-mnist/')))
from utils import mnist_reader


# hard coded fashion labels 0-9
FASHION_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

X_train, y_train = mnist_reader.load_mnist('../.data/fashion-mnist/data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('../.data/fashion-mnist/data/fashion', kind='t10k')

def show_samples(X, y, n=5, save_path=None):
    plt.figure(figsize=(10, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        label = FASHION_LABELS[y[i]] if y[i] < len(FASHION_LABELS) else str(y[i])
        plt.title(f"Label: {y[i]}: {label}")
        plt.axis('off')
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def test_load_mnist():
    assert X_train.shape == (60000, 784)
    assert y_train.shape == (60000,)
    assert X_test.shape == (10000, 784)
    assert y_test.shape == (10000,)
def test_data_values():
    assert X_train.min() >= 0 and X_train.max() <= 255
    assert X_test.min() >= 0 and X_test.max() <= 255
    assert y_train.min() >= 0 and y_train.max() <= 9
    assert y_test.min() >= 0 and y_test.max() <= 9  

main = __name__ == '__main__'
if main:
    test_load_mnist()
    test_data_values()
    print("All tests passed.")

    show_samples(X_test, y_test, n=5, save_path="./__fashion_samples.png")  # Show 5 sample images
else:
    print("Tests skipped. Run this script directly to execute tests.")