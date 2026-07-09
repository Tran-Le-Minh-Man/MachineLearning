import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config import Config

class Evaluator:
    def __init__(self):
        self.config = Config()

    def evaluate(self, model_name, y_true, y_pred):
        # Tính điểm chính xác tổng quát (Accuracy)
        acc = accuracy_score(y_true, y_pred)
        print(f"\n--- Kết quả {model_name} ---")
        print(f"Accuracy (Độ chính xác): {acc:.4f}")
        
        # In bảng báo cáo chi tiết gồm các chỉ số: Precision, Recall, F1-score cho từng loại quần áo
        print("Classification Report:\n", classification_report(y_true, y_pred, target_names=self.config.CLASS_NAMES))

        # Tính toán ma trận nhầm lẫn (Confusion Matrix) để xem model hay đoán sai ở những nhãn nào
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        
        # Vẽ biểu đồ nhiệt (Heatmap) trực quan cho ma trận nhầm lẫn
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.config.CLASS_NAMES,
                    yticklabels=self.config.CLASS_NAMES)
        plt.xlabel('Dự đoán')
        plt.ylabel('Thực tế')
        plt.title(f'Confusion Matrix - {model_name}')

        # Đường dẫn và tên file ảnh kết quả cần lưu
        save_path = os.path.join(self.config.OUTPUT_DIR, f'cm_{model_name.replace(" ", "_")}.png')
        plt.savefig(save_path)
        
        # Hiển thị hình vẽ ngay trên giao diện (như Google Colab/Kaggle)
        plt.show()
        
        # Đóng tất cả các hình vẽ để giải phóng bộ nhớ RAM và tránh hiển thị thông báo trống thừa
        plt.close('all')
