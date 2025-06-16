import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)

class PricingOptimizer:
    """
    负责执行最终的价格优化博弈、可视化结果，并分析模型决策依据。
    """
    def __init__(self, g_model, f_model_clf, f_model_reg, data: pd.DataFrame, large_order_threshold: int = 500):
        """
        初始化优化器。
        """
        self.g_model = g_model
        self.f_model_clf = f_model_clf
        self.f_model_reg = f_model_reg
        self.df = data.astype({'stockcode': 'category', 'customer_id': 'category'})
        self.LARGE_ORDER_THRESHOLD = large_order_threshold
        self._precompute_historical_stats()
        print("价格优化器已初始化，所有模型和数据已加载。")

    def _precompute_historical_stats(self):
        """
        预先计算一些历史统计数据。
        """
        print("正在预计算历史统计数据...")
        self.avg_prices = self.df.groupby('stockcode', observed=True)['price'].mean().to_dict()
        large_orders = self.df[self.df['quantity'] >= self.LARGE_ORDER_THRESHOLD]
        self.large_order_median_quantity = large_orders.groupby('stockcode', observed=True)['quantity'].median().to_dict()
        self.global_large_order_median = large_orders['quantity'].median()
        if pd.isna(self.global_large_order_median):
             self.global_large_order_median = self.LARGE_ORDER_THRESHOLD
        print("历史统计数据计算完成。")

    def run_price_optimization(self, top_n_customers: int = 10, top_n_products: int = 20):
        """
        为Top N的客户和Top N的商品执行价格优化。
        """
        print(f"\n--- 开始执行阶段三：为Top {top_n_customers} 客户和 Top {top_n_products} 商品进行价格优化 ---")
        
        product_df = self.df[self.df['stockcode'] != 'DOT']
        
        top_customers = self.df.groupby('customer_id', observed=True)['price'].sum().nlargest(top_n_customers).index
        top_products = product_df.groupby('stockcode', observed=True)['price'].sum().nlargest(top_n_products).index
        
        print(f"已选定 {len(top_customers)} 个客户和 {len(top_products)} 个商品（已排除邮费'DOT'）进行分析。")

        optimization_pairs = []
        for customer in top_customers:
            for product in top_products:
                optimization_pairs.append({'customer_id': customer, 'stockcode': product})
        
        if not optimization_pairs:
            print("警告: 未能生成客户-商品优化对。")
            return pd.DataFrame()

        sim_df = pd.DataFrame(optimization_pairs)
        recommendations = []
        
        self.training_features = self.g_model.feature_name_
        
        for _, row in tqdm(sim_df.iterrows(), total=sim_df.shape[0], desc="优化进度"):
            customer_id = row['customer_id']
            stockcode = row['stockcode']
            
            hist_price = self.avg_prices.get(stockcode, 1)
            base_features_row = self.df[(self.df['customer_id'] == customer_id) & (self.df['stockcode'] == stockcode)]
            
            if base_features_row.empty:
                customer_latest_trx = self.df[self.df['customer_id'] == customer_id].iloc[[-1]].copy()
                product_latest_trx = self.df[self.df['stockcode'] == stockcode].iloc[[-1]].copy()
                if customer_latest_trx.empty or product_latest_trx.empty: continue
                
                base_features_row = customer_latest_trx
                base_features_row['stockcode'] = stockcode
                for col in ['product_trade_count', 'product_rejection_rate']:
                     base_features_row[col] = product_latest_trx[col].iloc[0]
            else:
                base_features_row = base_features_row.iloc[[-1]].copy()

            best_price = hist_price
            max_revenue = 0
            
            price_range = np.linspace(max(0.01, hist_price * 0.7), hist_price * 1.5, 20)

            for p in price_range:
                test_case = base_features_row.copy()
                test_case['price'] = p
                test_case['price_deviation'] = (p - hist_price) / (hist_price + 1e-6)
                
                features_for_model = test_case[self.training_features]

                for col in ['stockcode', 'customer_id']:
                    original_categories = self.df[col].cat.categories
                    features_for_model[col] = pd.Categorical(features_for_model[col], categories=original_categories)
                
                if features_for_model.isnull().values.any(): continue

                prob_success = self.g_model.predict_proba(features_for_model)[:, 1][0]
                if prob_success < 0.2: continue

                prob_large_order = self.f_model_clf.predict_proba(features_for_model)[:, 1][0]
                pred_regular_qty_log = self.f_model_reg.predict(features_for_model)[0]
                pred_regular_qty = np.expm1(pred_regular_qty_log)
                
                estimated_large_qty = self.large_order_median_quantity.get(stockcode, self.global_large_order_median)
                expected_quantity = (1 - prob_large_order) * pred_regular_qty + prob_large_order * estimated_large_qty
                expected_revenue = p * prob_success * expected_quantity
                
                if expected_revenue > max_revenue:
                    max_revenue = expected_revenue
                    best_price = p

            recommendations.append({
                'customer_id': customer_id, 'stockcode': stockcode,
                'historical_avg_price': hist_price, 'recommended_price': best_price,
                'max_expected_revenue': max_revenue
            })
        print("价格优化全部完成。")
        return pd.DataFrame(recommendations)

    def visualize_and_analyze(self, recommendations_df: pd.DataFrame):
        """
        将优化结果可视化。
        """
        if recommendations_df.empty:
            print("没有可供分析的定价建议。")
            return
        print("\n--- 开始执行阶段四：结果可视化与分析 ---")
        
        recommendations_df = recommendations_df.sort_values('max_expected_revenue', ascending=False).head(15)
        
        print("\n[1/2] Top 15 定价策略推荐:")
        print(recommendations_df.to_string())

        print("\n[2/2] 正在生成价格对比图...")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('seaborn-v0_8-whitegrid')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        recs_plot = recommendations_df.copy()
        recs_plot['id'] = recs_plot['customer_id'].astype(str) + '-' + recs_plot['stockcode'].astype(str)
        x = np.arange(len(recs_plot))
        width = 0.4
        ax.bar(x - width/2, recs_plot['historical_avg_price'], width, label='Historical average price')
        ax.bar(x + width/2, recs_plot['recommended_price'], width, label='Recommend the best price')
        ax.set_title('Top 15 Comparison of Pricing Strategies for Product-Customer Combinations', fontsize=16, pad=20)
        ax.set_xlabel('Customer ID - Product ID', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(recs_plot['id'], rotation=60, ha="right")
        ax.legend()
        fig.tight_layout()
        plt.show()

    def analyze_feature_importance(self):
        """
        --- 新增方法 ---
        可视化三个核心模型的特征重要性，以理解模型决策依据。
        """
        print("\n--- 开始执行阶段五：模型可解释性分析 (特征重要性) ---")
        
        models = {
            "Transaction-success-rate model (g_model)": self.g_model,
            "Large-order classifier (f_model_clf)": self.f_model_clf,
            "Regular-order regressor (f_model_reg)": self.f_model_reg
        }
        
        for model_name, model in models.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            lgb.plot_importance(model, ax=ax, max_num_features=10, height=0.8, 
                                importance_type='gain', title=f'{model_name} Top 10 important features')
            ax.set_xlabel('Feature importance (Gain)', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            plt.tight_layout()
            plt.show()

        print("\n分析报告：")
        print("特征重要性图表揭示了每个模型在做决策时最看重的因素。")
        print("- 'Gain'（增益）类型的特征重要性，衡量了每个特征平均为模型带来的信息增益。值越高，说明该特征对于提升模型预测准确度的贡献越大。")
        print("- 通过观察这些图表，我们可以理解例如：究竟是`price`本身，还是`price_deviation`（价格偏离度），或是客户的历史行为 (`customer_cancellation_rate`) 对模型的决策影响更大。")

