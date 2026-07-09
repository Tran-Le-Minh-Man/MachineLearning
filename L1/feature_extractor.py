import numpy as np
from sklearn.decomposition import PCA
from config import Config

class FeatureExtractor:
    def __init__(self):
        self.config = Config()
        self.pca = None  # Lưu lại PCA đã fit để dùng cho tập test

    def fit_transform(self, X_train):
        """Fit và transform tập train. Phải gọi hàm này trước transform()."""
        method = self.config.FEATURE_METHOD.lower()

        if method == 'pca':
            # Fit PCA chỉ trên tập train để tránh data leakage từ tập test
            self.pca = PCA(n_components=self.config.PCA_COMPONENTS)
            return self.pca.fit_transform(X_train)

        elif method == 'hog':
            return self._extract_hog(X_train)

        elif method == 'raw':
            return X_train

        else:
            raise ValueError(f"Phương pháp '{method}' chưa được hỗ trợ.")

    def transform(self, X_test):
        """Transform tập test dùng các tham số đã fit từ tập train."""
        method = self.config.FEATURE_METHOD.lower()

        if method == 'pca':
            if self.pca is None:
                raise ValueError("Cần gọi fit_transform trước.")
            return self.pca.transform(X_test)

        elif method == 'hog':
            return self._extract_hog(X_test)

        elif method == 'raw':
            return X_test

    def _extract_hog(self, X):
        """Trích xuất đặc trưng HOG từ tập ảnh đầu vào."""
        from skimage.feature import hog
        features = []
        for img in X:
            # Reshape từ vector (784,) về ảnh 2D (28x28) trước khi tính HOG
            img_2d = img.reshape(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
            fd = hog(
                img_2d,
                orientations=self.config.HOG_ORIENTATIONS,
                pixels_per_cell=self.config.HOG_PIXELS_PER_CELL,
                cells_per_block=self.config.HOG_CELLS_PER_BLOCK,
                visualize=False
            )
            features.append(fd)
        return np.array(features)
