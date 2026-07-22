import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

class CustomerDataPreprocessor:
    """
    Handles RFM calculation, skewness correction, encoding, scaling,
    and feature preparation for clustering models.
    """
    def __init__(self, scaler_save_path: str = None):
        self.scaler_save_path = scaler_save_path
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_names = []
        self.fitted = False

    def compute_rfm_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes standard RFM quartiles/quintiles (1 to 5).
        R: 5 is most recent, 1 is least recent.
        F: 5 is highest frequency, 1 is lowest.
        M: 5 is highest monetary spend, 1 is lowest.
        """
        df_rfm = df.copy()
        
        df_rfm['R_Score'] = pd.qcut(df_rfm['Recency_Days'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
        df_rfm['F_Score'] = pd.qcut(df_rfm['Frequency_Orders'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        df_rfm['M_Score'] = pd.qcut(df_rfm['Monetary_Spend'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        
        df_rfm['RFM_Score_Sum'] = df_rfm['R_Score'] + df_rfm['F_Score'] + df_rfm['M_Score']
        df_rfm['RFM_Group'] = df_rfm['R_Score'].astype(str) + df_rfm['F_Score'].astype(str) + df_rfm['M_Score'].astype(str)
        
        return df_rfm

    def ensure_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if 'Avg_Order_Value' not in df_out.columns:
            if 'Monetary_Spend' in df_out.columns and 'Frequency_Orders' in df_out.columns:
                df_out['Avg_Order_Value'] = (df_out['Monetary_Spend'] / df_out['Frequency_Orders'].replace(0, 1)).round(2)
            else:
                df_out['Avg_Order_Value'] = 100.0
                
        if 'Churn_Risk_Index' not in df_out.columns:
            rec_days = df_out['Recency_Days'] if 'Recency_Days' in df_out.columns else 30
            freq_orders = df_out['Frequency_Orders'] if 'Frequency_Orders' in df_out.columns else 5
            eng_score = df_out['Engagement_Score'] if 'Engagement_Score' in df_out.columns else 50
            support_tix = df_out['Support_Tickets'] if 'Support_Tickets' in df_out.columns else 1
            
            rec_factor = (rec_days / 120.0).clip(upper=1.0)
            freq_factor = (1.0 - (freq_orders / 20.0)).clip(lower=0.0)
            eng_factor = (1.0 - (eng_score / 100.0)).clip(lower=0.0)
            support_factor = (support_tix / 5.0).clip(upper=1.0)
            df_out['Churn_Risk_Index'] = (0.4 * rec_factor + 0.3 * freq_factor + 0.2 * eng_factor + 0.1 * support_factor).round(3)
            
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Fits scaler and encoder, and transforms training features.
        """
        df_processed = self.compute_rfm_scores(df)
        df_processed = self.ensure_engineered_features(df_processed)
        
        # Numerical features for clustering
        num_cols = [
            'Recency_Days', 'Frequency_Orders', 'Monetary_Spend', 
            'Avg_Order_Value', 'Category_Diversity', 'Engagement_Score', 
            'Discount_Ratio', 'Return_Rate', 'Churn_Risk_Index'
        ]
        
        # Categorical features
        cat_cols = ['Preferred_Channel', 'Gender']
        
        # Log transform skewed metrics
        df_log = df_processed[num_cols].copy()
        df_log['Monetary_Spend'] = np.log1p(df_log['Monetary_Spend'])
        df_log['Frequency_Orders'] = np.log1p(df_log['Frequency_Orders'])
        df_log['Recency_Days'] = np.log1p(df_log['Recency_Days'])
        df_log['Avg_Order_Value'] = np.log1p(df_log['Avg_Order_Value'])
        
        scaled_num = self.scaler.fit_transform(df_log)
        encoded_cat = self.encoder.fit_transform(df_processed[cat_cols])
        
        cat_feature_names = self.encoder.get_feature_names_out(cat_cols).tolist()
        self.feature_names = num_cols + cat_feature_names
        
        X_matrix = np.hstack([scaled_num, encoded_cat])
        self.fitted = True
        
        if self.scaler_save_path:
            os.makedirs(os.path.dirname(self.scaler_save_path), exist_ok=True)
            joblib.dump({
                'scaler': self.scaler,
                'encoder': self.encoder,
                'feature_names': self.feature_names,
                'num_cols': num_cols,
                'cat_cols': cat_cols
            }, self.scaler_save_path)
            print(f"Saved preprocessing pipeline to {self.scaler_save_path}")

        return X_matrix, df_processed

    def transform_single_customer(self, customer_dict: dict, pipeline_path: str = None) -> np.ndarray:
        """
        Transforms a single raw customer input dictionary into scaled model matrix.
        """
        if not self.fitted and pipeline_path and os.path.exists(pipeline_path):
            pipeline = joblib.load(pipeline_path)
            self.scaler = pipeline['scaler']
            self.encoder = pipeline['encoder']
            self.feature_names = pipeline['feature_names']
            num_cols = pipeline['num_cols']
            cat_cols = pipeline['cat_cols']
            self.fitted = True
        else:
            num_cols = [
                'Recency_Days', 'Frequency_Orders', 'Monetary_Spend', 
                'Avg_Order_Value', 'Category_Diversity', 'Engagement_Score', 
                'Discount_Ratio', 'Return_Rate', 'Churn_Risk_Index'
            ]
            cat_cols = ['Preferred_Channel', 'Gender']

        df_single = pd.DataFrame([customer_dict])
        if 'Avg_Order_Value' not in df_single.columns:
            df_single['Avg_Order_Value'] = (df_single['Monetary_Spend'] / df_single['Frequency_Orders']).round(2)
        if 'Churn_Risk_Index' not in df_single.columns:
            df_single['Churn_Risk_Index'] = (
                0.50 * (df_single['Recency_Days'] / 365.0) +
                0.25 * (1.0 - (df_single['Engagement_Score'] / 100.0)) +
                0.15 * (df_single['Support_Tickets'] / 12.0) +
                0.10 * (df_single['Return_Rate'] / 0.30)
            ).clip(0.0, 1.0).round(4)
            
        df_log = df_single[num_cols].copy()
        df_log['Monetary_Spend'] = np.log1p(df_log['Monetary_Spend'])
        df_log['Frequency_Orders'] = np.log1p(df_log['Frequency_Orders'])
        df_log['Recency_Days'] = np.log1p(df_log['Recency_Days'])
        df_log['Avg_Order_Value'] = np.log1p(df_log['Avg_Order_Value'])
        
        scaled_num = self.scaler.transform(df_log)
        encoded_cat = self.encoder.transform(df_single[cat_cols])
        
        return np.hstack([scaled_num, encoded_cat])
