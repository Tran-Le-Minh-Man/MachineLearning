import os

class Config:
    """
    Nơi tập trung toàn bộ tham số của dự án.
    Muốn thay đổi bất kỳ thứ gì thì chỉ cần sửa ở đây, không cần đụng vào code logic.
    """

    PROJECT_NAME = "Fashion MNIST - Phân loại trang phục"

    # Ảnh Fashion MNIST mặc định là grayscale 28x28 pixel, 1 kênh màu
    IMAGE_SIZE = 28
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # Chia 80% để train, 20% để test — tỷ lệ phổ biến trong ML
    VAL_SPLIT_RATIO = 0.2
    # Cố định seed để mỗi lần chạy cho cùng một kết quả, dễ so sánh mô hình
    RANDOM_STATE = 42

    # Chọn phương pháp rút trích đặc trưng: 'pca', 'hog' hoặc 'raw'
    # SVM và LR không chạy tốt trên pixel thô nên bắt buộc phải rút trích trước
    FEATURE_METHOD = 'pca'

    # Số chiều giữ lại sau PCA (từ 784 chiều xuống còn 100)
    PCA_COMPONENTS = 100

    # Tham số HOG: chia ảnh thành các ô 8x8, mỗi ô có 9 hướng gradient
    HOG_ORIENTATIONS = 9
    HOG_PIXELS_PER_CELL = (8, 8)
    HOG_CELLS_PER_BLOCK = (2, 2)

    # Tham số Logistic Regression
    # C là nghịch đảo của độ mạnh regularization — C càng nhỏ thì regularization càng mạnh
    LOGISTIC_C = 1.0
    LOGISTIC_MAX_ITER = 1000  # Tăng số vòng lặp để đảm bảo hội tụ

    # Danh sách kernel SVM cần thử nghiệm để so sánh hiệu năng
    # linear: nhanh, phù hợp dữ liệu tuyến tính
    # rbf: chậm hơn nhưng xử lý tốt hơn dữ liệu phi tuyến
    SVM_KERNELS = ['linear', 'rbf']
    SVM_C = 1.0
    SVM_GAMMA = 'scale'  # Tự tính gamma dựa trên số chiều và phương sai dữ liệu

    # Thư mục lưu ảnh kết quả (Confusion Matrix)
    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
