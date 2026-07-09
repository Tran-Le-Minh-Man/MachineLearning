from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from models import ModelsTrainer
from evaluator import Evaluator

def main():
    # 1. Tải và chuẩn bị dữ liệu (Tự động tải online, chuẩn hóa và chia tập Train/Test theo tỷ lệ 80/20)
    loader = DataLoader()
    X_train, X_test, y_train, y_test = loader.load_data()

    # 2. Rút trích đặc trưng (Mặc định dùng PCA để giảm số chiều từ 784 xuống còn 100)
    extractor = FeatureExtractor()
    X_train_features = extractor.fit_transform(X_train)
    X_test_features = extractor.transform(X_test)

    # Khởi tạo các module huấn luyện và đánh giá
    trainer = ModelsTrainer()
    evaluator = Evaluator()

    # 3. Huấn luyện và đánh giá mô hình Logistic Regression
    logistic_model = trainer.train_logistic(X_train_features, y_train)
    y_pred_logistic = trainer.predict(logistic_model, X_test_features)
    evaluator.evaluate("Logistic Regression", y_test, y_pred_logistic)

    # 4. Huấn luyện và đánh giá mô hình SVM với các kernel khác nhau (Linear, RBF)
    svm_kernels = trainer.config.SVM_KERNELS
    for kernel in svm_kernels:
        svm_model = trainer.train_svm(X_train_features, y_train, kernel=kernel)
        y_pred_svm = trainer.predict(svm_model, X_test_features)
        evaluator.evaluate(f"SVM (Kernel: {kernel})", y_test, y_pred_svm)

if __name__ == "__main__":
    main()
