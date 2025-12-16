import json, glob, pandas as pd, numpy as np, re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import joblib

def parse_one(path):
    """解析单个JSON文件"""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]  # 统一成列表

    rows = []
    for turn in data:
        is_scam = 0 if turn.get("riskType") == "无风险" else 1
        
        rows.append({
            "text": turn.get("text", ""),
            "is_scam": is_scam,
            "timestamp": turn.get("timestamp", ""),
            "riskType": turn.get("riskType", ""),
        })
    return pd.DataFrame(rows)

# 1. 加载数据
print("加载数据...")
files = glob.glob("D:\\1111AAAAAA课\\社会计算\\FGRC-SCD-dialog\\dialog\\prompt1_8w_selected.json")
df = pd.concat([parse_one(f) for f in tqdm(files, desc="Parsing")], ignore_index=True)

# 2. 文本清洗
def clean(txt):
    """清洗文本"""
    txt = re.sub(r"http\S+|www\.\S+", " ", txt)
    txt = re.sub(r"@\w+", " ", txt)
    txt = re.sub(r"\[.*?\]", " ", txt)   # 表情
    txt = re.sub(r'【.*?】', ' ', txt)    # 去掉【风险点】
    txt = re.sub(r'riskType|riskPoint', ' ', txt, flags=re.I)
    return txt.lower()

df["text_clean"] = df.text.astype(str).apply(clean)

# 3. 构造对话特征
print("构造对话特征...")

# 3.1 对话长度
df["dialog_length"] = df["text_clean"].apply(lambda x: len(x.split()))

# 3.2 关键词特征
def extract_keywords(text):
    keywords = ['账号', '退款', '转账', '冻结', '客服', '提现', '支付', '转钱', '金额']
    return sum([1 for word in keywords if word in text])

df["keyword_count"] = df["text_clean"].apply(extract_keywords)

# 3.3 提及金额特征
def extract_amount(text):
    # 提取金额：匹配类似“3700.2元”这种金额表达方式
    match = re.search(r'(\d+(?:\.\d+)?)\s?元', text)
    if match:
        return float(match.group(1))
    return 0

df["amount_mentioned"] = df["text_clean"].apply(extract_amount)

# 4. 构造社会信号特征（对话特征）
dialog_cols = ['dialog_length', 'keyword_count', 'amount_mentioned']

# 5. 划分训练测试集
print("\n划分训练测试集...")
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.is_scam)
print(f"训练集: {len(train)}, 测试集: {len(test)}")
print(f"训练集中诈骗比例: {train.is_scam.mean():.3f}")
print(f"测试集中诈骗比例: {test.is_scam.mean():.3f}")

# 6. 构建预处理管道
print("\n构建特征处理管道...")
text_col = "text_clean"

# 6.1 仅文本特征（用于对比）
text_only_preprocessor = TfidfVectorizer(
    max_features=500,  # 500维
    ngram_range=(1, 2),
    min_df=5
)

# 6.2 文本+对话特征
combined_preprocessor = ColumnTransformer([
    ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=5), text_col),
    ("dialog", StandardScaler(), dialog_cols)  # 标准化对话特征
])

# 7. 训练模型
print("\n训练模型...")

# 7.1 仅文本模型（对比基准）
print("训练仅文本模型...")
text_only_model = Pipeline([
    ("vectorizer", text_only_preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ))
])
text_only_model.fit(train[text_col], train.is_scam)

# 7.2 文本+对话特征模型
print("训练文本+对话特征模型...")
combined_model = Pipeline([
    ("preprocessor", combined_preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ))
])
combined_model.fit(train, train.is_scam)

# 8. 评估模型
print("\n评估模型...")

# 8.1 仅文本模型预测
text_only_test_proba = text_only_model.predict_proba(test[text_col])[:, 1]
text_only_auc = roc_auc_score(test.is_scam, text_only_test_proba)

# 8.2 文本+对话特征模型预测
combined_test_proba = combined_model.predict_proba(test)[:, 1]
combined_auc = roc_auc_score(test.is_scam, combined_test_proba)

# 8.3 打印结果
print("\n" + "="*50)
print("模型性能对比:")
print("="*50)
print(f"仅文本模型 (500维TF-IDF): AUC = {text_only_auc:.4f}")
print(f"文本+对话特征模型: AUC = {combined_auc:.4f}")
print(f"AUC提升: {combined_auc - text_only_auc:.4f}")
print("="*50)

# 9. 详细分类报告
print("\n仅文本模型分类报告:")
print(classification_report(test.is_scam, (text_only_test_proba >= 0.5).astype(int)))

print("\n文本+对话特征模型分类报告:")
print(classification_report(test.is_scam, (combined_test_proba >= 0.5).astype(int)))

# 10. 特征重要性分析
print("\n特征重要性分析...")
# 获取特征名称
tfidf_features = combined_model.named_steps['preprocessor'].named_transformers_['tfidf'].get_feature_names_out()
all_features = list(tfidf_features) + dialog_cols
print(f"总特征数: {len(all_features)} (TF-IDF: {len(tfidf_features)}, 对话特征: {len(dialog_cols)})")

# 获取模型系数
if hasattr(combined_model.named_steps['classifier'], 'coef_'):
    coefficients = combined_model.named_steps['classifier'].coef_[0]
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'coefficient': coefficients,
        'abs_coefficient': abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\nTop 20重要特征:")
    print(feature_importance.head(20))
    
    print("\n对话特征排名:")
    dialog_importance = feature_importance[feature_importance['feature'].isin(dialog_cols)]
    print(dialog_importance)

# 11. 保存结果
print("\n保存结果...")

# 保存预测结果
pred_df = pd.DataFrame({
    "text": test["text"].values,
    "true_label": test.is_scam.values,
    "text_only_prob": text_only_test_proba,
    "combined_prob": combined_test_proba,
    "text_only_pred": (text_only_test_proba >= 0.5).astype(int),
    "combined_pred": (combined_test_proba >= 0.5).astype(int)
})
pred_df.to_csv("predictions.csv", index=False, encoding='utf-8-sig')

# 保存模型
joblib.dump(text_only_model, "text_only_model.pkl")
joblib.dump(combined_model, "combined_model.pkl")

# 12. 生成实验报告
report = f"""
# 金融诈骗检测实验报告

## 1. 数据概况
- 总样本数：{len(df):,}
- 诈骗样本比例：{df.is_scam.mean():.2%}
- 训练集大小：{len(train):,}
- 测试集大小：{len(test):,}

## 2. 特征工程
### 2.1 文本特征
- 清洗：移除URL、@提及、表情符号等
- TF-IDF：500维，1-2 gram

### 2.2 对话特征（共{len(dialog_cols)}个）
1. 对话长度 (dialog_length)
2. 关键词数量 (keyword_count)
3. 提及金额 (amount_mentioned)

## 3. 实验结果
| 模型 | AUC | 精确率 | 召回率 | F1分数 |
|------|-----|--------|--------|--------|
| 仅文本 | {text_only_auc:.4f} | - | - | - |
| 文本+对话特征 | {combined_auc:.4f} | - | - | - |

**AUC提升：{combined_auc - text_only_auc:.4f}**

## 4. 结论
{'对话特征对金融诈骗检测有明显提升效果' if combined_auc > text_only_auc else '对话特征在本实验中提升效果有限'}

## 5. 文件输出
1. `predictions.csv`：测试集预测结果
2. `text_only_model.pkl`：仅文本模型
3. `combined_model.pkl`：文本+对话特征模型
"""

with open("experiment_report.md", "w", encoding="utf-8") as f:
    f.write(report)

print("\n" + "="*50)
print("实验完成!")
print(f"报告已保存到: experiment_report.md")
print(f"预测结果已保存到: predictions.csv")
print("="*50)
