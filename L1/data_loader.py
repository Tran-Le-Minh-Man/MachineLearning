import numpy as np
from sklearn.model_selection import train_test_split
from config import Config

class DataLoader:
    def __init__(self):
        self.config = Config()

    def load_data(self):
        try:
            # Ưu tiên Keras vì nhanh hơn và luôn có sẵn trên Colab/Kaggle
            from tensorflow.keras.datasets import fashion_mnist
            (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = fashion_mnist.load_data()

            # Gộp train và test gốc lại để chia lại theo tỷ lệ 80/20 của nhóm
            pixels = np.concatenate((X_train_raw, X_test_raw), axis=0)
            labels = np.concatenate((y_train_raw, y_test_raw), axis=0)

            # Dàn phẳng ảnh (28x28) thành vector 1 chiều (784,)
            pixels = pixels.reshape(pixels.shape[0], -1)

        except ImportError:
            # Dự phòng nếu máy chưa cài Keras/TensorFlow
            from sklearn.datasets import fetch_openml
            dataset = fetch_openml('Fashion-MNIST', version=1, cache=True, parser='auto')
            pixels = dataset.data.values
            labels = dataset.target.astype(int).values

        # Chuẩn hóa pixel về [0, 1] để SVM và LR hội tụ ổn định hơn
        pixels = pixels.astype(np.float32) / 255.0

        # stratify=labels đảm bảo tỷ lệ mỗi lớp được giữ nguyên sau khi chia
        X_train, X_test, y_train, y_test = train_test_split(
            pixels,
            labels,
            test_size=self.config.VAL_SPLIT_RATIO,
            random_state=self.config.RANDOM_STATE,
            stratify=labels
        )

        return X_train, X_test, y_train, y_test
