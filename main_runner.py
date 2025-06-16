import os
import pandas as pd

# --- 模块导入 ---
try:
    from data_preprocessing_feature_engeneering import AdvancedDataPreprocessor
    print("成功从 'data_preprocessing_feature_engeneering.py' 导入 AdvancedDataPreprocessor。")
except ImportError:
    print("错误: 无法导入 AdvancedDataPreprocessor...")
    exit()

try:
    from model_building import DynamicPricingModeler
    print("成功从 'model_building.py' 导入 DynamicPricingModeler。")
except ImportError:
    print("错误: 无法导入 DynamicPricingModeler...")
    exit()

try:
    from price_optimizer import PricingOptimizer
    print("成功从 'price_optimizer.py' 导入 PricingOptimizer。")
except ImportError:
    print("错误: 无法导入 PricingOptimizer...")
    exit()

def main():
    """
    项目的主执行函数，按顺序协调整个流程。
    """
    print("\n" + "="*50)
    print("开始执行动态定价分析完整流程...")
    print("="*50 + "\n")

    processed_data_path = 'final_processed_data.csv'
    
    # --- 阶段一：数据预处理 ---
    if os.path.exists(processed_data_path):
        print(f"发现已处理的数据文件 '{processed_data_path}'，直接加载。")
        final_processed_df = pd.read_csv(processed_data_path, 
                                         parse_dates=['invoicedate'],
                                         dtype={'stockcode': 'category', 'customer_id': 'category'})
    else:
        # (如果文件不存在，则执行数据预处理)
        print("\n>>> 正在执行阶段一：数据预处理...")
        # ... (数据预处理代码) ...
    
    # --- 阶段二：模型评估 ---
    print("\n>>> 正在执行阶段二：构建并评估模型...")
    modeler_for_eval = DynamicPricingModeler(processed_data=final_processed_df)
    modeler_for_eval.build_follower_response_models()
    print("\n阶段二模型性能评估完成。")
    
    # --- 阶段三：价格优化 ---
    print("\n>>> 正在执行阶段三：价格优化...")
    print("正在使用全部数据重新训练最终模型以用于优化...")
    final_modeler = DynamicPricingModeler(processed_data=final_processed_df)
    final_g, final_f_clf, final_f_reg = final_modeler.train_final_models()
    print("最终模型训练完成。")

    optimizer = PricingOptimizer(
        g_model=final_g, f_model_clf=final_f_clf, f_model_reg=final_f_reg,
        data=final_processed_df
    )
    recommendations = optimizer.run_price_optimization()
    
    # --- 阶段四：结果可视化 ---
    optimizer.visualize_and_analyze(recommendations)
    
    # --- 新增：阶段五：模型可解释性分析 ---
    optimizer.analyze_feature_importance()

    print("\n" + "="*50)
    print("动态定价分析完整流程全部成功执行！")
    print("="*50)

if __name__ == '__main__':
    main()

