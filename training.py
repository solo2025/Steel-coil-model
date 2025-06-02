import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import os

# 读取数据
train_data = pd.read_csv('normal_trainset.csv')
test_data = pd.read_csv('normal_testset.csv')

# 提取特征和目标变量
features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19']
X_train = train_data[features]
y_train = train_data['y1_noisy']
X_test = test_data[features]
y_test = test_data['y1_noisy']

# 定义基类
class BaseModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.results = {
            'model_name': model_name,
            'train_score': None,
            'train_mae': None,
            'test_score': None,
            'test_mae': None
        }

    def train(self, X_train, y_train):
        raise NotImplementedError

    def save_model(self, path):
        joblib.dump(self.model, path)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def calculate_metrics(self, y_true, y_pred):
        score = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return score, mae

    def save_results(self, path):
        results_df = pd.DataFrame([self.results])
        results_df.to_csv(path, index=False)

    def plot_predictions(self, y_train_true, y_train_pred, y_test_true, y_test_pred, path):
        plt.figure(figsize=(10, 6))
        
        # 训练集：真实值 vs 预测值
        plt.scatter(y_train_true, y_train_pred, alpha=0.5, color='blue', label='Train Set')
        
        # 测试集：真实值 vs 预测值
        plt.scatter(y_test_true, y_test_pred, alpha=0.5, color='red', label='Test Set')
        
        # 添加完美预测参考线（y = x）
        min_val = min(min(y_train_true), min(y_test_true))
        max_val = max(max(y_train_true), max(y_test_true))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
        
        # 添加标签和标题
        plt.xlabel('True Values (y1)')
        plt.ylabel('Predicted Values')
        plt.title(f'{self.model_name}: True vs Predicted (Train & Test)')
        plt.legend()
        plt.grid(True)
        
        # 保存图像
        plt.savefig(path)
        plt.close()

# 1. AdaBoost
class AdaBoostModel(BaseModel):
    def __init__(self):
        super().__init__('AdaBoost')
        self.model = AdaBoostRegressor(random_state=42)

    def train(self, X_train, y_train, n_estimators=50, learning_rate=0.1):
        self.model.set_params(n_estimators=n_estimators, learning_rate=learning_rate)
        self.model.fit(X_train, y_train)
        self.results['hyperparameters'] = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate
        }

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        if X_train is not None and y_train is not None:
            train_pred = self.predict(X_train)
            self.results['train_score'], self.results['train_mae'] = self.calculate_metrics(y_train, train_pred)
        
        test_pred = self.predict(X_test)
        self.results['test_score'], self.results['test_mae'] = self.calculate_metrics(y_test, test_pred)

# 2. Gradient Boosting (GBM)
class GBMModel(BaseModel):
    def __init__(self):
        super().__init__('GradientBoosting')
        self.model = GradientBoostingRegressor(random_state=42)

    def train(self, X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.model.set_params(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        self.model.fit(X_train, y_train)
        self.results['hyperparameters'] = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth
        }

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        if X_train is not None and y_train is not None:
            train_pred = self.predict(X_train)
            self.results['train_score'], self.results['train_mae'] = self.calculate_metrics(y_train, train_pred)
        
        test_pred = self.predict(X_test)
        self.results['test_score'], self.results['test_mae'] = self.calculate_metrics(y_test, test_pred)

# 3. XGBoost
class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__('XGBoost')
        self.model = xgb.XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='mae')

    def train(self, X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3, gamma=0, reg_lambda=1):
        self.model.set_params(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            gamma=gamma,
            reg_lambda=reg_lambda
        )
        self.model.fit(X_train, y_train)
        self.results['hyperparameters'] = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'gamma': gamma,
            'reg_lambda': reg_lambda
        }

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        if X_train is not None and y_train is not None:
            train_pred = self.predict(X_train)
            self.results['train_score'], self.results['train_mae'] = self.calculate_metrics(y_train, train_pred)
        
        test_pred = self.predict(X_test)
        self.results['test_score'], self.results['test_mae'] = self.calculate_metrics(y_test, test_pred)

# 4. LightGBM
class LightGBMModel(BaseModel):
    def __init__(self):
        super().__init__('LightGBM')
        self.model = lgb.LGBMRegressor(random_state=47)

    def train(self, X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=-1, num_leaves=31, min_data_in_leaf=20):
        self.model.set_params(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_data_in_leaf=min_data_in_leaf
        )
        self.model.fit(X_train, y_train)
        self.results['hyperparameters'] = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_data_in_leaf': min_data_in_leaf
        }

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        if X_train is not None and y_train is not None:
            train_pred = self.predict(X_train)
            self.results['train_score'], self.results['train_mae'] = self.calculate_metrics(y_train, train_pred)
        
        test_pred = self.predict(X_test)
        self.results['test_score'], self.results['test_mae'] = self.calculate_metrics(y_test, test_pred)

# 5. CatBoost
class CatBoostModel(BaseModel):
    def __init__(self):
        super().__init__('CatBoost')
        # 初始化时设置参数（避免后续修改）
        self.model = CatBoostRegressor(verbose=0, iterations=100, learning_rate=0.1, depth=6, l2_leaf_reg=3)

    def train(self, X_train, y_train, iterations=100, learning_rate=0.1, depth=6, l2_leaf_reg=3, random_state=42):
        # 重新初始化模型（避免参数冲突）
        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_state=random_state,
            verbose=0
        )
        self.model.fit(X_train, y_train)
        self.results['hyperparameters'] = {
            'iterations': iterations,
            'learning_rate': learning_rate,
            'depth': depth,
            'l2_leaf_reg': l2_leaf_reg,
            'random_state': random_state
        }

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        if X_train is not None and y_train is not None:
            train_pred = self.predict(X_train)
            self.results['train_score'], self.results['train_mae'] = self.calculate_metrics(y_train, train_pred)
        
        test_pred = self.predict(X_test)
        self.results['test_score'], self.results['test_mae'] = self.calculate_metrics(y_test, test_pred)

# 示例：训练并评估所有模型
if __name__ == '__main__':
    # 初始化模型
    models = {
        # 'LightGBM': LightGBMModel(),
        'CatBoost': CatBoostModel(),
        # 'XGBoost': XGBoostModel(),
        # 'GBM': GBMModel(),
        # 'AdaBoost': AdaBoostModel()
    }

    # 训练和评估
    for name, model in models.items():
        if name == 'AdaBoost':
            model.train(X_train, y_train, n_estimators=50, learning_rate=0.1)
        elif name == 'GBM':
            model.train(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3)
        elif name == 'XGBoost':
            model.train(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3)
        elif name == 'LightGBM':
            model.train(X_train, y_train, n_estimators=2000, learning_rate=0.3, max_depth=7, num_leaves=63, min_data_in_leaf=30)
        elif name == 'CatBoost':
            iterations = [5000]
            learning_rates = [0.4]
            depths = [6]
            l2_leaf_regs = [1]
            _rs = []
            _iterations = []
            _learning_rates = []
            _depths = []
            _l2_leaf_regs = []
            train_scores = []
            test_scores = []
            train_maes = []
            test_maes = []

            for rs in [45]:
                for iteration in iterations:
                    for learning_rate in learning_rates:
                        for depth in depths:
                            for l2_leaf_reg in l2_leaf_regs:
                                # 重新初始化模型（避免参数冲突）
                                model.model = CatBoostRegressor(
                                    iterations=iteration,
                                    learning_rate=learning_rate,
                                    depth=depth,
                                    l2_leaf_reg=l2_leaf_reg,
                                    random_state=rs,
                                    verbose=0
                                )
                                model.model.fit(X_train, y_train)
                                if not os.path.exists('output'):
                                    os.makedirs('output')
                                model.save_model(f'output/catboost_iteration{iteration}_lr{learning_rate}_depth{depth}_l2{l2_leaf_reg}_rs{rs}.pkl')
                                y_train_pred = model.predict(X_train)
                                y_test_pred = model.predict(X_test)
                                train_score, train_mae = model.calculate_metrics(y_train, y_train_pred)
                                test_score, test_mae = model.calculate_metrics(y_test, y_test_pred)
                                # plot
                                model.plot_predictions(
                                    y_train_true=y_train,
                                    y_train_pred=y_train_pred,
                                    y_test_true=y_test,
                                    y_test_pred=y_test_pred,
                                    path=f'output/catboost_plot_iteration{iteration}_lr{learning_rate}_depth{depth}_l2{l2_leaf_reg}_rs{rs}.png'
                                )
                                _rs.append(rs)
                                _iterations.append(iteration)
                                _learning_rates.append(learning_rate)
                                _depths.append(depth)
                                _l2_leaf_regs.append(l2_leaf_reg)
                                train_scores.append(train_score)
                                test_scores.append(test_score)
                                train_maes.append(train_mae)
                                test_maes.append(test_mae)

                                print(f"Model: {name}, Iteration: {iteration}, Learning Rate: {learning_rate}, Depth: {depth}, L2 Leaf Reg: {l2_leaf_reg}, Random State: {rs}")
                                print(f"Train Score: {train_score:.4f}, Train MAE: {train_mae:.4f}")
                                print(f"Test Score: {test_score:.4f}, Test MAE: {test_mae:.4f}")
            results_df = pd.DataFrame({
                'iterations': _iterations,
                'learning_rate': _learning_rates,
                'depth': _depths,
                'l2_leaf_reg': _l2_leaf_regs,
                'train_score': train_scores,
                'test_score': test_scores,
                'train_mae': train_maes,
                'test_mae': test_maes
            })
            results_df.to_csv('catboost_results.csv', index=False)