import pandas as pd
import numpy as np
import duckdb
import warnings
import os

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class AdvancedDataPreprocessor:
    """
    一个数据预处理器模块，修正了数据泄露问题，
    采用时间感知的方法进行特征工程。
    """
    def __init__(self, files: list, db_path: str = ':memory:'):
        self.files = files
        self.db_path = db_path
        self.con = None
        print(f"初始化数据预处理器，将使用DuckDB ({self.db_path})。")

    def _connect_db(self):
        self.con = duckdb.connect(database=self.db_path, read_only=False)
        print("DuckDB连接已建立。")

    def _load_and_initial_clean(self) -> pd.DataFrame:
        print(f"正在从 {len(self.files)} 个文件加载数据...")
        try:
            df_list = [pd.read_csv(file, encoding='utf-8') for file in self.files]
        except UnicodeDecodeError:
            print("UTF-8解码失败，尝试使用 ISO-8859-1 (Latin-1) 编码...")
            df_list = [pd.read_csv(file, encoding='ISO-8859-1') for file in self.files]

        df = pd.concat(df_list, ignore_index=True)
        print("数据加载完成。")

        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        required_cols = ['invoicedate', 'customer_id', 'quantity', 'price', 'invoice', 'stockcode']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"错误: CSV文件中缺少关键列 '{col}'。")

        df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
        df = df.dropna(subset=['invoicedate', 'customer_id'])
        df['customer_id'] = df['customer_id'].astype(float).astype(int).astype(str)
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

        df = df.dropna(subset=['quantity', 'price'])
        df = df[~((df['quantity'] <= 0) & (~df['invoice'].str.startswith('C', na=False)))]
        df = df[df['price'] >= 0]
        
        print(f"初步清洗完成。数据集大小: {df.shape}")
        return df

    def _classify_events_with_duckdb(self, df: pd.DataFrame) -> pd.DataFrame:
        print("正在使用DuckDB进行高级事件分类...")
        self.con.register('transactions', df)
        final_df = self.con.execute(self.get_duckdb_query()).fetchdf()
        print(f"DuckDB事件分类完成。筛选后数据集大小: {final_df.shape}")
        return final_df

    def _finalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        采用时间感知的方法，创建基础及高级聚合特征，防止数据泄露。
        """
        print("正在创建最终特征 (时间感知方法)...")
        
        # 1. 必须首先按时间排序，这是时间感知特征工程的基础
        df = df.sort_values('invoicedate').reset_index(drop=True)

        # 2. 创建基础特征
        df['is_success'] = np.where(df['invoice'].str.startswith('C', na=False), 0, 1)
        df['quantity'] = df['quantity'].abs()
        df['year'] = df['invoicedate'].dt.year
        df['month'] = df['invoicedate'].dt.month
        df['weekday'] = df['invoicedate'].dt.weekday
        df['hour'] = df['invoicedate'].dt.hour
        
        # --- 修正：创建无数据泄露的高级特征 ---
        print("正在创建无数据泄露的高级聚合特征...")
        
        # a. 客户维度特征 (使用扩展窗口)
        # groupby().shift(1) 是为了确保我们使用的是截至“上一次”交易的信息
        # expanding() 会计算从开始到当前行的所有数据的聚合值
        gb_customer = df.groupby('customer_id')
        df['customer_trade_count'] = gb_customer.cumcount()
        
        # 计算截至上一次交易的成功率
        cum_success = gb_customer['is_success'].cumsum()
        cum_count = gb_customer.cumcount() + 1
        df['customer_success_rate'] = (cum_success - df['is_success']) / (cum_count - 1)
        df['customer_cancellation_rate'] = 1 - df['customer_success_rate']

        # b. 商品维度特征
        gb_stock = df.groupby('stockcode')
        df['product_trade_count'] = gb_stock.cumcount()
        cum_success_prod = gb_stock['is_success'].cumsum()
        cum_count_prod = gb_stock.cumcount() + 1
        df['product_success_rate'] = (cum_success_prod - df['is_success']) / (cum_count_prod - 1)
        df['product_rejection_rate'] = 1 - df['product_success_rate']

        # c. 价格偏离度特征
        cum_price_sum = gb_stock['price'].cumsum()
        df['avg_product_price_so_far'] = (cum_price_sum - df['price']) / (cum_count_prod - 1)
        df['price_deviation'] = (df['price'] - df['avg_product_price_so_far']) / (df['avg_product_price_so_far'] + 1e-6)
        
        # d. 填充新增特征产生的缺失值（对于每个分组的第一次出现）
        df.fillna(0, inplace=True)
        
        print("高级聚合特征创建完成。")
        # -----------------------------

        # 2. 转换类别特征类型
        df['stockcode'] = df['stockcode'].astype('category')
        df['customer_id'] = df['customer_id'].astype('category')
        
        # 3. 清理不再需要的列
        df = df.drop(columns=['avg_product_price_so_far'], errors='ignore')
        
        print("特征工程完成。")
        return df

    def process(self) -> pd.DataFrame:
        self._connect_db()
        try:
            initial_df = self._load_and_initial_clean()
            classified_df = self._classify_events_with_duckdb(initial_df)
            final_df = self._finalize_features(classified_df)
        finally:
            if self.con:
                self.con.close()
                print("DuckDB连接已关闭。预处理流程结束。")
        return final_df
        
    def get_duckdb_query(self) -> str:
        # 辅助函数，返回完整的DuckDB查询字符串
        return """
        WITH
        tagged_transactions AS (
            SELECT *, CASE WHEN invoice LIKE 'C%' THEN 'negative' ELSE 'positive' END AS transaction_type, ROW_NUMBER() OVER () as unique_id
            FROM transactions
        ),
        matched_candidates AS (
            SELECT
                neg.unique_id AS neg_id, pos.unique_id AS pos_id,
                (epoch(neg.invoicedate) - epoch(pos.invoicedate)) / 3600.0 AS delta_hours,
                abs(neg.quantity) / pos.quantity AS return_ratio,
                abs(neg.quantity + pos.quantity) AS qty_diff
            FROM tagged_transactions AS neg JOIN tagged_transactions AS pos
            ON neg.customer_id = pos.customer_id AND neg.stockcode = pos.stockcode
            AND neg.transaction_type = 'negative' AND pos.transaction_type = 'positive'
            AND neg.invoicedate > pos.invoicedate
        ),
        best_match AS (
            SELECT neg_id, pos_id, delta_hours, return_ratio
            FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY neg_id ORDER BY delta_hours ASC, qty_diff ASC) as rn FROM matched_candidates) WHERE rn = 1
        ),
        classified_transactions AS (
            SELECT
                t.unique_id,
                CASE
                    WHEN bm.delta_hours < 24 AND bm.return_ratio >= 0.95 THEN 'price_rejection'
                    WHEN bm.delta_hours >= 24 AND bm.return_ratio >= 0.95 THEN 'quality_return'
                    WHEN bm.neg_id IS NOT NULL THEN 'other_return'
                    WHEN t.transaction_type = 'negative' AND bm.neg_id IS NULL THEN 'price_rejection'
                    ELSE 'positive'
                END AS event_type,
                CASE
                    WHEN bm.neg_id IS NOT NULL AND NOT (bm.delta_hours < 24 AND bm.return_ratio >= 0.95) THEN bm.pos_id
                    ELSE NULL
                END AS paired_pos_id_to_remove
            FROM tagged_transactions AS t LEFT JOIN best_match AS bm ON t.unique_id = bm.neg_id
        )
        SELECT t.* 
        FROM tagged_transactions AS t
        JOIN (
            SELECT unique_id FROM classified_transactions
            WHERE event_type = 'positive'
            AND unique_id NOT IN (
                SELECT paired_pos_id_to_remove 
                FROM classified_transactions 
                WHERE paired_pos_id_to_remove IS NOT NULL
            )
            UNION ALL
            SELECT unique_id FROM classified_transactions
            WHERE event_type = 'price_rejection'
        ) AS final_ids ON t.unique_id = final_ids.unique_id;       
        """
