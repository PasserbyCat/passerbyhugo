---
title: "Binary Prediction of Poisonous Mushrooms"
date: 2024-08-24T09:11:00+08:00
categories: ["blog"]
tags: ["數據", "報告"]
author: "passerbycat"
draft: false
---

## 動機  
我最一開始的動機相當簡單，就是kaggle上的獎勵，實在有點讓人心動，於是我便選擇了這個挑戰作為我的報告主題。  
![plot01](/images/毒菇分析/plot01.png)  

## 資料集分析  
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna
from sklearn.metrics import matthews_corrcoef
import scipy
import warnings
warnings.filterwarnings('ignore')  # 忽略警告訊息

# 讀取訓練數據集、測試數據集和樣本提交文件
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
sample_submission_data = pd.read_csv("sample_submission.csv")

# 輸出數據集的形狀 (行數, 列數)
print("train_data :", train_data.shape)
print("test_data :", test_data.shape)
print("sample_submission_data :", sample_submission_data.shape)
```

{{< admonition type=quote title="output" open=false >}}
train_data : (3116945, 22)  
test_data : (2077964, 21)  
sample_submission_data : (2077964, 2)  
{{< /admonition >}}

```python
# 顯示訓練數據集各列的數據類型及其非空值的數量
train_data.info()
```

{{< admonition type=quote title="output" open=false >}}
![plot02](/images/毒菇分析/plot02.png) 
{{< /admonition >}}

```python
# 計算並顯示每個特徵（欄位）中缺失值的數量
msno.bar(train_data)
```

{{< admonition type=quote title="output" open=false >}}
![plot07](/images/毒菇分析/plot07.png) 
{{< /admonition >}}

```python
# 計算並打印目標變量（類別）的每個類別的樣本數量
print(train_data['class'].value_counts())

# 繪製目標變量的類別分佈條形圖
sns.countplot(x='class', data=train_data)

# 旋轉X軸上的標籤，使其便於閱讀
plt.xticks(rotation=60)

# 顯示圖表
plt.show()
```

{{< admonition type=quote title="output" open=false >}}
![plot03](/images/毒菇分析/plot03.png) 
{{< /admonition >}}

```python
# 選擇數據集中所有類別型變量（object類型）的列
cate_col = train_data.select_dtypes(include=['object']).columns

# 計算每個類別型變量中每個類別的樣本數量
unique_categories = {col: train_data[col].value_counts() for col in cate_col}

# 設置整個圖形的大小，高度根據類別型變量的數量動態調整
plt.figure(figsize=(15, len(cate_col) * 5))

# 依次為每個類別型變量繪製條形圖
for i, (col, counts) in enumerate(unique_categories.items(), 1):
    # 創建一個子圖，根據變量數量動態調整位置
    plt.subplot(len(cate_col), 1, i)
    
    # 使用Seaborn繪製條形圖，X軸為類別名稱，Y軸為對應的樣本數量
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    
    # 設置每個子圖的標題，包括變量名稱
    plt.title(f"Count of unique categories in column '{col}'")
    
    # 設置X軸標籤
    plt.xlabel('Categories')
    
    # 設置Y軸標籤
    plt.ylabel('Count')
    
    # 旋轉X軸上的標籤，並向右對齊，以防止重疊
    plt.xticks(rotation=45, ha='right')
    
    # 自動調整子圖參數以避免重疊
    plt.tight_layout()

# 顯示整個圖形
plt.show()
```

{{< admonition type=quote title="output" open=false >}}
![plot04](/images/毒菇分析/plot04.png) 
{{< /admonition >}}

## 資料集處理
```python
train_data = train_data.drop(['id'], axis=1)
test_data = test_data.drop(['id'], axis=1)
```

```python
def cleaning(df):
    threshold = 100  # 設置頻率閾值，低於此閾值的類別將被標記為 'noise'
    
    # 定義需要處理的類別型變量名稱列表
    cat_feats = ["cap-shape", "cap-surface", "cap-color", "does-bruise-or-bleed", 
                 "gill-attachment", "gill-spacing", "gill-color", "stem-root", 
                 "stem-surface", "stem-color", "has-ring", "ring-type", 
                 "spore-print-color", "habitat", "season", "veil-color", "veil-type"]
    
    for feat in cat_feats:
        # 如果變量的類型為類別型，則進行以下處理
        if df[feat].dtype.name == 'category':
            # 如果 'missing' 不在該變量的類別中，則將 'missing' 類別添加進去
            if 'missing' not in df[feat].cat.categories:
                df[feat] = df[feat].cat.add_categories('missing')
            # 如果 'noise' 不在該變量的類別中，則將 'noise' 類別添加進去
            if 'noise' not in df[feat].cat.categories:
                df[feat] = df[feat].cat.add_categories('noise')
        else:
            # 如果變量不是類別型，則將其轉換為類別型，並添加 'missing' 和 'noise' 類別
            df[feat] = df[feat].astype('category')
            df[feat] = df[feat].cat.add_categories(['missing', 'noise'])
        
        # 將變量中的缺失值填充為 'missing'
        df[feat] = df[feat].fillna('missing')
        
        # 計算每個類別的頻率，將頻率低於閾值的類別標記為 'noise'
        counts = df[feat].value_counts(dropna=False)
        infrequent_categories = counts[counts < threshold].index
        df[feat] = df[feat].apply(lambda x: 'missing' if x in infrequent_categories else x)
    
    # 返回清理過後的數據框
    return df

# 對訓練集和測試集進行清理
train_data = cleaning(train_data)
test_data = cleaning(test_data)
```

```python
# 定義填充缺失值的函數
def fill_na_with_group_mean(data, group_by_features, target_feature):
    # 計算根據 group_by_features 分組後的 target_feature 的平均值
    group_means = data.groupby(group_by_features)[target_feature].mean()
    
    def fill_na(row):
        # 如果 target_feature 在當前行中為缺失值，則根據分組平均值進行填充
        if pd.isna(row[target_feature]):
            group = tuple(row[group_by_features])  # 獲取分組鍵
            return group_means.get(group, np.nan)  # 從 group_means 中查找分組的平均值
        else:
            return row[target_feature]  # 如果不為缺失值，則返回原值
    
    # 應用填充函數，填充缺失值
    data[target_feature] = data.apply(fill_na, axis=1)
    # 如果仍有缺失值，用眾數進行填充
    data[target_feature].fillna(data[target_feature].mode()[0], inplace=True)

# 對訓練集和測試集分別應用填充函數
fill_na_with_group_mean(train_data, ['stem-width', 'stem-height'], 'cap-diameter')
fill_na_with_group_mean(test_data, ['stem-width', 'stem-height'], 'cap-diameter')

# 填充測試集中的 'stem-height' 的缺失值
test_data['stem-height'].fillna(test_data['stem-height'].mode()[0], inplace=True)
```

```python
cat_feats = ["cap-shape", "cap-surface", "cap-color", "does-bruise-or-bleed", "gill-attachment", 
             "gill-spacing", "gill-color", "stem-root", "stem-surface", "stem-color", 
             "has-ring", "ring-type", "spore-print-color", "habitat", "season", 
             "veil-color", "veil-type"]

# 將所有類別型特徵轉換為category類型
for feat in cat_feats:
    train_data[feat] = train_data[feat].astype('category')
    test_data[feat] = test_data[feat].astype('category')
```

```python
train_data.isna().sum()
```

{{< admonition type=quote title="output" open=false >}}
![plot05](/images/毒菇分析/plot05.png) 
{{< /admonition >}}

```python
# 將特徵與目標變數分開
X = train_data.drop(['class'], axis=1)  # 刪除 'class' 列，保留所有其他列作為特徵
y = train_data['class']  # 將 'class' 列作為目標變數

# 對目標變數進行編碼
label_encoder = LabelEncoder()  
y = label_encoder.fit_transform(y)  # 將類別型的 'class' 變數轉換為數值型標籤

# 將數據分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
```

## 參數調整
```python
def objective(trial):
    # 定義一個目標函數，用於在 Optuna 的調參過程中測試不同的參數組合
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # 樹的數量範圍在 100 到 1000 之間
        'max_depth': trial.suggest_int('max_depth', 3, 20),  # 樹的最大深度範圍在 3 到 20 之間
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),  # 學習率的範圍是 0.01 到 0.3
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),  # 每棵樹隨機採樣的比例，範圍是 0.5 到 1.0
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),  # 每棵樹使用特徵的比例，範圍是 0.5 到 1.0
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),  # 最小損失減少值的範圍，避免過度擬合
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),  # L2 正則化項的範圍
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),  # L1 正則化項的範圍
        'scale_pos_weight': trial.suggest_uniform('scale_pos_weight', 1.0, 10.0),  # 用於處理不平衡資料的正負樣本比例
        'use_label_encoder': False,  # 關閉 XGBoost 自帶的標籤編碼
        'eval_metric': 'logloss',  # 評估指標使用對數損失（Logarithmic Loss）
        'enable_categorical': True  # 啟用對類別型特徵的支持
    }

    # 創建 XGBoost 模型並根據當前 trial 的參數進行訓練
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # 使用模型進行預測，並計算 Matthews correlation coefficient (MCC)
    y_pred = model.predict(X_test)
    return matthews_corrcoef(y_test, y_pred)

# 創建一個 Optuna 的 study 對象，用來進行參數優化，並設置優化方向為最大化
study = optuna.create_study(direction='maximize')

# 開始優化，設置進行 100 次試驗
study.optimize(objective, n_trials=100)

# 輸出最佳參數
print(f"Best parameters: {study.best_params}")
```

## 模型分析
```python
# 設定 XGBoost 模型的參數
parameters = {
    'n_estimators': 297,  # 樹的數量
    'max_depth': 16,  # 樹的最大深度
    'learning_rate': 0.03906159386409017,  # 學習率
    'subsample': 0.6935900010487451,  # 用於訓練每顆樹的資料比例
    'colsample_bytree': 0.5171160704967471,  # 用於訓練每顆樹的特徵比例
    'gamma': 0.00013710778966124443,  # 用於減少過擬合的最小損失減益
    'lambda': 0.0017203271581656767,  # L2 正則化項
    'alpha': 8.501510750413265e-06,  # L1 正則化項
    'scale_pos_weight': 1.0017942891559255,  # 正負樣本的權重比例
    'enable_categorical': True,  # 啟用類別特徵處理
    'tree_method': 'hist',  # 使用直方圖加速的樹生成方法
    'device': 'cuda'  # 在 GPU 上進行訓練
}

# 初始化 XGBoost 分類器並進行模型訓練
model = XGBClassifier(**parameters)
model = model.fit(X, y)

# 直接使用模型預測並創建提交的 DataFrame
submission_df = pd.DataFrame({
    'id': sample_submission_data['id'],  # 直接從 sample_submission_data 中獲取 'id'
    'class': np.where(model.predict(test_data) > 0.5, 'p', 'e')  # 直接生成 'class' 列，將預測結果轉換為 'p' 或 'e'
})

# 最終上傳文件
submission_df.to_csv('submission.csv', index=False)
```

{{< admonition type=quote title="output" open=false >}}
![plot06](/images/毒菇分析/plot06.png) 
{{< /admonition >}}