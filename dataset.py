import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

def inv_log1p(x):  # обратно из лога в рубли
    return np.expm1(x)

# Target Encoder
class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, n_splits=5, smoothing=20.0, random_state=42, target_col='price_doc'):
        self.cols = cols; self.n_splits = n_splits; self.smoothing = smoothing
        self.random_state = random_state; self.target_col = target_col
        self.global_mean_ = None; self.mappings_ = {}; self._fitted_on_rows_ = None

    def fit(self, X, y=None):
        df = X.copy()
        if y is None and self.target_col in df.columns: y = df[self.target_col].values
        else: df[self.target_col] = y
        df = df[~pd.isna(df[self.target_col])].copy()
        self._fitted_on_rows_ = len(df)
        if df.empty:
            self.global_mean_ = 0.0; self.mappings_ = {c: {} for c in self.cols}; return self
        self.global_mean_ = df[self.target_col].mean()
        self.mappings_.clear()
        for col in self.cols:
            stats = df.groupby(col, dropna=False)[self.target_col].agg(['mean','count'])
            smooth = (stats['count']*stats['mean'] + self.smoothing*self.global_mean_) / (stats['count']+self.smoothing)
            self.mappings_[col] = smooth.to_dict()
        return self

    def transform(self, X, y=None, use_oof_if_possible=False):
        df = X.copy()
        if use_oof_if_possible and (y is not None) and (len(df) == self._fitted_on_rows_):
            df_oof = df.copy(); df_oof[self.target_col] = y
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            for col in self.cols:
                oof = pd.Series(index=df_oof.index, dtype=float)
                for tr_idx, va_idx in kf.split(df_oof):
                    tr = df_oof.iloc[tr_idx]
                    stats = tr.groupby(col)[self.target_col].agg(['mean','count'])
                    smooth = (stats['count']*stats['mean'] + self.smoothing*self.global_mean_) / (stats['count']+self.smoothing)
                    oof.iloc[va_idx] = df_oof.iloc[va_idx][col].map(smooth).fillna(self.global_mean_)
                df[col] = oof.values
            return df
        for col in self.cols:
            df[col] = df[col].map(self.mappings_.get(col, {})).fillna(self.global_mean_)
        return df

# Препроцессор
class HousingPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, low_card_max=20, kfold_splits=5, smoothing=20.0,
                 target_col='price_doc', target_log=True,
                 clip_quantiles=(0.01, 0.99),
                 log_cols=('full_sq','life_sq','kitch_sq','mkad_km','ttk_km','sadovoe_km','kremlin_km')):
        self.low_card_max=low_card_max; self.kfold_splits=kfold_splits; self.smoothing=smoothing
        self.target_col=target_col; self.target_log=target_log
        self.clip_quantiles=clip_quantiles; self.log_cols=set(log_cols)
        self.num_cols_=None; self.low_card_cols_=None; self.high_card_cols_=None
        self.ohe_=None; self.te_=None; self.columns_out_=None; self.global_medians_={}; self.cat_fill_value_="Unknown"

    @staticmethod
    def _parse_dates(df):
        if 'timestamp' in df.columns:
            df = df.copy(); df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    @staticmethod
    def _merge_macro(left_df, macro_df):
        if macro_df is None: return left_df
        macro = macro_df.copy(); macro['timestamp']=pd.to_datetime(macro['timestamp'])
        base = left_df.copy(); base['timestamp']=pd.to_datetime(base['timestamp'])
        return base.merge(macro, on='timestamp', how='left', suffixes=('', '_macro'))

    @staticmethod
    def _basic_fixes(df):
        df = df.copy()
        for c in ['full_sq','life_sq','kitch_sq']:
            if c in df.columns: df.loc[df[c] <= 0, c] = np.nan
        if {'life_sq','full_sq'}.issubset(df.columns): df.loc[df['life_sq'] > df['full_sq'], 'life_sq'] = np.nan
        if 'kitch_sq' in df.columns and 'full_sq' in df.columns: df.loc[df['kitch_sq'] > df['full_sq'], 'kitch_sq'] = np.nan
        if 'build_year' in df.columns: df.loc[(df['build_year']<1850)|(df['build_year']>2050),'build_year']=np.nan
        return df

    @staticmethod
    def _date_features(df):
        df = df.copy()
        if 'timestamp' in df.columns:
            df['year']=df['timestamp'].dt.year; df['month']=df['timestamp'].dt.month
            df['quarter']=df['timestamp'].dt.quarter; df['is_dec_jan']=df['month'].isin([12,1]).astype(int)
        return df

    @staticmethod
    def _domain_features(df):
        df = df.copy()
        if {'floor','max_floor'}.issubset(df.columns):
            df['floor_ratio']=df['floor']/df['max_floor'].replace({0:np.nan})
        if {'life_sq','full_sq'}.issubset(df.columns):
            df['life_ratio']=df['life_sq']/df['full_sq']
        if {'kitch_sq','full_sq'}.issubset(df.columns):
            df['kitch_ratio']=df['kitch_sq']/df['full_sq']
        if {'build_year','timestamp'}.issubset(df.columns):
            year=df['timestamp'].dt.year; df['house_age']=year-df['build_year']
        return df

    def _clip_skewed(self, df):
        df = df.copy()
        if self.clip_quantiles:
            ql, qh = self.clip_quantiles
            for c in df.select_dtypes(include=np.number).columns:
                if df[c].notna().sum()==0: continue
                lo, hi = df[c].quantile(ql), df[c].quantile(qh)
                if pd.isna(lo) or pd.isna(hi): continue
                df[c] = df[c].clip(lower=lo, upper=hi)
        return df

    def _log_transform(self, df):
        df = df.copy()
        for c in self.log_cols:
            if c in df.columns: df[c] = np.log1p(df[c])
        return df

    def _split_column_types(self, df):
        cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in num_cols: num_cols.remove(self.target_col)
        return cat_cols, num_cols

    def fit(self, df, macro_df=None, y=None):
        df=self._parse_dates(df); df=self._merge_macro(df, macro_df); df=self._basic_fixes(df)
        df=self._date_features(df); df=self._domain_features(df); df=self._clip_skewed(df); df=self._log_transform(df)
        if y is None and self.target_col in df.columns: y = df[self.target_col].values
        cats, nums = self._split_column_types(df)
        for c in nums:
            med=df[c].median(); self.global_medians_[c]=med; df[c]=df[c].fillna(med)
        for c in cats: df[c]=df[c].fillna(self.cat_fill_value_)
        self.low_card_cols_=[c for c in cats if df[c].nunique()<=20]
        self.high_card_cols_=[c for c in cats if df[c].nunique()>20]
        try: self.ohe_=OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError: self.ohe_=OneHotEncoder(handle_unknown='ignore', sparse=False)
        if self.low_card_cols_: self.ohe_.fit(df[self.low_card_cols_])
        self.te_=KFoldTargetEncoder(self.high_card_cols_, n_splits=5, smoothing=20.0, target_col=self.target_col)
        if self.high_card_cols_: self.te_.fit(df, y)
        self.num_cols_=[c for c in nums]
        ohe_names=self.ohe_.get_feature_names_out(self.low_card_cols_).tolist() if self.low_card_cols_ else []
        self.columns_out_=self.num_cols_+ohe_names+self.high_card_cols_
        return self

    def transform(self, df, macro_df=None, y=None, return_target=True, for_train_oof=False):
        df=self._parse_dates(df); df=self._merge_macro(df, macro_df); df=self._basic_fixes(df)
        df=self._date_features(df); df=self._domain_features(df); df=self._clip_skewed(df); df=self._log_transform(df)
        if return_target and (y is None) and (self.target_col in df.columns): y=df[self.target_col].values
        cats, nums = self._split_column_types(df)
        for c in nums:
            med=self.global_medians_.get(c, df[c].median()); df[c]=df[c].fillna(med)
        for c in cats: df[c]=df[c].fillna(self.cat_fill_value_)
        for c in self.num_cols_:
            if c not in df.columns: df[c]=0.0
        X_num = df[self.num_cols_].values
        X_ohe = self.ohe_.transform(df[self.low_card_cols_]) if self.low_card_cols_ else np.empty((len(df),0))
        if self.high_card_cols_:
            if for_train_oof and (y is not None):
                X_te_df = self.te_.transform(df, y=y, use_oof_if_possible=True)
            else:
                X_te_df = self.te_.transform(df)
            for c in self.high_card_cols_:
                if c not in X_te_df.columns: X_te_df[c]=self.te_.global_mean_
            X_te = X_te_df[self.high_card_cols_].values
        else:
            X_te = np.empty((len(df),0))
        X = np.hstack([X_num, X_ohe, X_te])
        if return_target:
            y_out = np.log1p(y) if (y is not None) else None  # учим модель на логе
            return X, y_out
        else:
            return X


train = pd.read_csv('train.csv')
macro = pd.read_csv('macro.csv')

split_date = '2015-06-01'
mask_tr = pd.to_datetime(train['timestamp']) < pd.to_datetime(split_date)

pp = HousingPreprocessor(target_log=True)
pp.fit(train[mask_tr], macro_df=macro)

# train
X_tr, y_tr_log = pp.transform(train[mask_tr], macro_df=macro, for_train_oof=True)
# test/val (mapping)
X_te, y_te_log = pp.transform(train[~mask_tr], macro_df=macro, for_train_oof=False)

# модель
model = HistGradientBoostingRegressor(
    loss='squared_error', learning_rate=0.06, max_depth=8, max_bins=255,
    min_samples_leaf=40, l2_regularization=0.1, early_stopping=True,
    validation_fraction=0.1, random_state=42
)
model.fit(X_tr, y_tr_log)


pred_tr_log = model.predict(X_tr)
rmse_train = np.sqrt(mean_squared_error(np.expm1(y_tr_log), np.expm1(pred_tr_log))) / 1_000_000
print(f"RMSE (train): {rmse_train:.3f} млн ₽")


pred_te_log = model.predict(X_te)
rmse_test = np.sqrt(mean_squared_error(np.expm1(y_te_log), np.expm1(pred_te_log))) / 1_000_000
print(f"RMSE (test):  {rmse_test:.3f} млн ₽")
