# 다양한 모델을 predict_proba_pos로 통일하는 래퍼 - 김민규 작성

import numpy as np

class ModelAdapter:
    def __init__(self, model):
        self.model = model

    def predict_proba_pos(self, X):
        m = self.model
        if hasattr(m, "predict_proba"):
            return m.predict_proba(X)[:, 1]
        if hasattr(m, "decision_function"):
            s = m.decision_function(X)
            return 1.0 / (1.0 + np.exp(-s))
        y = m.predict(X).astype(float)
        return 0.01 + 0.98*y