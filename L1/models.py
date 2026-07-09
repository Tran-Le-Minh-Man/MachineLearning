from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from config import Config

class ModelsTrainer:
    def __init__(self):
        self.config = Config()

    def train_logistic(self, X_train, y_train):
        # n_jobs=-1: dùng toàn bộ CPU cores để tăng tốc huấn luyện
        model = LogisticRegression(
            C=self.config.LOGISTIC_C,
            max_iter=self.config.LOGISTIC_MAX_ITER,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    def train_svm(self, X_train, y_train, kernel='linear'):
        # gamma='scale': tự động tính gamma = 1 / (n_features * X.var())
        model = SVC(
            C=self.config.SVM_C,
            kernel=kernel,
            gamma=self.config.SVM_GAMMA,
            random_state=self.config.RANDOM_STATE
        )
        model.fit(X_train, y_train)
        return model

    def predict(self, model, X_test):
        return model.predict(X_test)
