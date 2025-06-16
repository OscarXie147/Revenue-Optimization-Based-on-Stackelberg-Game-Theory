import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

class DynamicPricingModeler:
    """
    负责构建、训练和评估动态定价所需的混合模型。
    """
    def __init__(self, processed_data: pd.DataFrame, large_order_threshold: int = 500):
        if not isinstance(processed_data, pd.DataFrame) or processed_data.empty:
            raise ValueError("需要提供一个非空的、经过预处理的Pandas DataFrame。")
            
        self.df = processed_data
        self.LARGE_ORDER_THRESHOLD = large_order_threshold
        print(f"“大额订单”阈值设定为: quantity >= {self.LARGE_ORDER_THRESHOLD}")

        g_success_count = self.df['is_success'].value_counts().get(1, 1)
        g_rejection_count = self.df['is_success'].value_counts().get(0, 1)
        g_scale_weight = g_success_count / g_rejection_count
        self.g_model = lgb.LGBMClassifier(random_state=42, objective='binary', metric='auc', scale_pos_weight=g_scale_weight)
        
        self.f_model_clf = lgb.LGBMClassifier(random_state=42, objective='binary', metric='auc')
        self.f_model_reg = lgb.LGBMRegressor(random_state=42, objective='regression_l1', metric='rmse')
        
        print("动态定价建模器已初始化，准备构建混合模型。")

    def build_follower_response_models(self):
        print("\n--- 开始构建阶段二：跟随者响应模型 ---")
        
        # 更新：使用新的、无数据泄露的特征列表
        features = [
            'price', 'stockcode', 'customer_id', 'month', 'weekday', 'hour',
            'customer_trade_count', 'customer_cancellation_rate',
            'product_trade_count', 'product_rejection_rate', 'price_deviation'
        ]
        categorical_features = ['stockcode', 'customer_id']
        
        # --- 模型一 (g_model): 交易成功率分类模型 ---
        print("\n[1/3] 正在构建 g_model (交易成功率分类模型)...")
        train_df = self.df[self.df['year'] < 2011]
        test_df = self.df[self.df['year'] == 2011]
        X_train_g, y_train_g = train_df[features], train_df['is_success']
        X_test_g, y_test_g = test_df[features], test_df['is_success']
        self.g_model.fit(X_train_g, y_train_g, categorical_feature=categorical_features,
                         eval_set=[(X_test_g, y_test_g)], callbacks=[lgb.early_stopping(10, verbose=False)])
        y_pred_proba_g = self.g_model.predict_proba(X_test_g)[:, 1]
        auc_score_g = roc_auc_score(y_test_g, y_pred_proba_g)
        print(f"g_model 在测试集上的评估完成。ROC-AUC Score: {auc_score_g:.4f}")

        # --- F-Model 混合策略 ---
        success_df = self.df[self.df['is_success'] == 1].copy()
        success_df['is_large_order'] = (success_df['quantity'] >= self.LARGE_ORDER_THRESHOLD).astype(int)
        
        train_success_df = success_df[success_df['year'] < 2011]
        test_success_df = success_df[success_df['year'] == 2011]

        # --- 模型二 (f_model_clf): 大额订单分类器 ---
        print("\n[2/3] 正在构建 f_model_clf (大额订单分类器)...")
        target_f_clf = 'is_large_order'
        X_train_f_clf, y_train_f_clf = train_success_df[features], train_success_df[target_f_clf]
        X_test_f_clf, y_test_f_clf = test_success_df[features], test_success_df[target_f_clf]
        
        clf_pos_count = y_train_f_clf.value_counts().get(0, 1)
        clf_neg_count = y_train_f_clf.value_counts().get(1, 1)
        clf_scale_weight = clf_pos_count / clf_neg_count
        self.f_model_clf.set_params(scale_pos_weight=clf_scale_weight)
        
        self.f_model_clf.fit(X_train_f_clf, y_train_f_clf, categorical_feature=categorical_features,
                             eval_set=[(X_test_f_clf, y_test_f_clf)], callbacks=[lgb.early_stopping(10, verbose=False)])
        
        y_pred_proba_f_clf = self.f_model_clf.predict_proba(X_test_f_clf)[:, 1]
        auc_score_f_clf = roc_auc_score(y_test_f_clf, y_pred_proba_f_clf)
        print(f"f_model_clf 在测试集上的评估完成。ROC-AUC Score: {auc_score_f_clf:.4f}")

        # --- 模型三 (f_model_reg): 常规订单回归器 ---
        print("\n[3/3] 正在构建 f_model_reg (常规订单回归器)...")
        
        regular_orders_train = train_success_df[train_success_df['is_large_order'] == 0].copy()
        regular_orders_test = test_success_df[test_success_df['is_large_order'] == 0].copy()
        
        regular_orders_train['quantity_log'] = np.log1p(regular_orders_train['quantity'])
        
        X_train_f_reg, y_train_f_reg = regular_orders_train[features], regular_orders_train['quantity_log']
        X_test_f_reg, y_test_f_reg_actual = regular_orders_test[features], regular_orders_test['quantity']

        self.f_model_reg.fit(X_train_f_reg, y_train_f_reg, categorical_feature=categorical_features,
                             eval_set=[(X_test_f_reg, np.log1p(y_test_f_reg_actual))],
                             callbacks=[lgb.early_stopping(10, verbose=False)])

        y_pred_log_f_reg = self.f_model_reg.predict(X_test_f_reg)
        y_pred_f_reg = np.expm1(y_pred_log_f_reg)
        y_pred_f_reg[y_pred_f_reg < 0] = 0
        
        rmse_reg = np.sqrt(mean_squared_error(y_test_f_reg_actual, y_pred_f_reg))
        print(f"f_model_reg 在【常规订单】测试集上的评估完成。RMSE: {rmse_reg:.4f}")

        print("\n--- 阶段二模型构建全部完成 ---")
        return self.g_model, self.f_model_clf, self.f_model_reg
    def train_final_models(self):
        """
        在全部数据集上训练最终的模型，不进行划分，用于最终的优化。
        """
        print("在全部数据上训练最终模型...")

        features = [
            'price', 'stockcode', 'customer_id', 'month', 'weekday', 'hour',
            'customer_trade_count', 'customer_cancellation_rate',
            'product_trade_count', 'product_rejection_rate', 'price_deviation'
        ] 
        categorical_features = ['stockcode', 'customer_id']

        # 训练最终的 g_model
        X_g, y_g = self.df[features], self.df['is_success']
        self.g_model.fit(X_g, y_g, categorical_feature=categorical_features)
        print("最终 g_model 训练完成。")

        # 准备f-model的数据
        success_df = self.df[self.df['is_success'] == 1].copy()
        success_df['is_large_order'] = (success_df['quantity'] >= self.LARGE_ORDER_THRESHOLD).astype(int)
        
        # 训练最终的 f_model_clf
        X_f_clf, y_f_clf = success_df[features], success_df['is_large_order']
        self.f_model_clf.fit(X_f_clf, y_f_clf, categorical_feature=categorical_features)
        print("最终 f_model_clf 训练完成。")

        # 训练最终的 f_model_reg
        regular_orders = success_df[success_df['is_large_order'] == 0].copy()
        regular_orders['quantity_log'] = np.log1p(regular_orders['quantity'])
        X_f_reg, y_f_reg = regular_orders[features], regular_orders['quantity_log']
        self.f_model_reg.fit(X_f_reg, y_f_reg, categorical_feature=categorical_features)
        print("最终 f_model_reg 训练完成。")

        return self.g_model, self.f_model_clf, self.f_model_reg