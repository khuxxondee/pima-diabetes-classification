# Pima Indians Diabetes Classification

A complete machine learning project to predict diabetes using the Pima Indians Diabetes dataset from UCI / Kaggle — built step by step with focus on understanding every decision made.

**Goal:** Predict whether a patient has diabetes (binary classification: 0 = No Diabetes, 1 = Diabetes)

**Dataset:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
768 patients · 8 features · 1 target variable

---

## Pipeline Overview

| Step | Task | Key output |
|------|------|------------|
| 1 | Load & explore | ทำความเข้าใจโครงสร้างข้อมูล |
| 2 | Detect missing values | พบ hidden missing values |
| 3 | Imputation | จัดการค่าที่หายไป |
| 4 | EDA & Visualization | วิเคราะห์ความสัมพันธ์ |
| 5 | Feature engineering | ปรับ scale ข้อมูล |
| 6 | Train/test split | แบ่งข้อมูล 80/20 |
| 7 | Modeling | เปรียบเทียบ 3 โมเดล |
| 8 | Evaluation | เลือกโมเดลที่เหมาะสม |
| 9 | Enhanced visualization | Heatmap + Confusion matrix |
| 10 | Cross validation | ตรวจสอบความน่าเชื่อถือ |
| 11 | Feature importance | วิเคราะห์ความสำคัญของ features |
| 12 | GridSearchCV | Hyperparameter tuning |
| 13 | sklearn Pipeline | รวมทุกขั้นตอนเป็น production-ready code |

---

## Step-by-step Details

### Step 1 — Load & Explore
โหลดข้อมูลด้วย `pd.read_csv()` แล้วตรวจสอบโครงสร้างด้วย `df.info()` และ `df.describe()`

**สิ่งที่พบ:**
- 768 แถว, 9 columns (8 features + 1 target)
- ไม่มี NaN ที่ pandas มองเห็น → แต่นั่นไม่ใช่ความจริง (พบใน Step 2)

---

### Step 2 — Detect Missing Values
ค่า 0 ใน 5 columns ต่อไปนี้ไม่สมเหตุสมผลทางการแพทย์:
`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`

**วิธีตรวจจับ:** ใช้ `df.describe()` ดูค่า min — ถ้าความดันโลหิตเป็น 0 คนนั้นไม่มีชีวิตแล้ว  
**วิธีแก้:** แปลง 0 → `NaN` เพื่อให้ pandas มองเห็นเป็น missing value

```python
df['Glucose'] = df['Glucose'].replace(0, np.nan)
```

**ผลที่ได้ (% missing):**
| Column | Missing % |
|--------|-----------|
| Insulin | 48.7% |
| SkinThickness | 29.6% |
| BloodPressure | 4.6% |
| BMI | 1.4% |
| Glucose | 0.7% |

---

### Step 3 — Imputation
จัดการค่าที่หายไปตามสัดส่วน:

- **Glucose, BloodPressure, BMI** (< 5%) → `dropna()` ตัดแถวทิ้ง เพราะเสียน้อยมาก
- **Insulin, SkinThickness** (> 20%) → `fillna(median)` เติมด้วย median เพราะมี outlier เยอะ (median ไม่ถูก outlier กระทบ)

**ผลลัพธ์:** ข้อมูลเหลือ 724 แถว, ไม่มี NaN เลย

---

### Step 4 — EDA & Visualization
วิเคราะห์ความสัมพันธ์ระหว่าง features กับ Outcome ด้วย Boxplot และ Correlation

**Boxplot:** เปรียบเทียบ Insulin ระหว่างกลุ่มเป็น/ไม่เป็นเบาหวาน  
→ กลุ่มเป็นเบาหวานมี median Insulin สูงกว่าชัดเจน

**Correlation กับ Outcome:**
| Feature | Correlation |
|---------|-------------|
| Glucose | 0.49 ← สูงที่สุด |
| BMI | 0.30 |
| Age | 0.25 |
| Insulin | 0.21 |

**Key insight:** ไม่มี feature ไหน correlate สูง เป็นเรื่องปกติสำหรับข้อมูลสุขภาพ ML จะรวมทุก feature เพื่อชดเชย

---

### Step 5 — Feature Engineering
ใช้ `StandardScaler` ทำให้ทุก feature อยู่ใน scale เดียวกัน (mean=0, std=1)

**ทำไมต้อง scale:** Insulin มีช่วง 0-800 แต่ Pregnancies มีช่วง 0-17 ถ้าไม่ scale โมเดลจะ bias ไปหา feature ที่ตัวเลขใหญ่กว่า

```python
X = df.drop(columns=['Outcome'])  # 8 features
y = df['Outcome']                  # target
X_scaled = StandardScaler().fit_transform(X)
```

---

### Step 6 — Train/Test Split
แบ่งข้อมูล 80/20 ด้วย `train_test_split`

```
724 แถว → Train: 579 แถว (80%) + Test: 145 แถว (20%)
```

**`random_state=42`** เพื่อให้ผลการสุ่มเหมือนเดิมทุกครั้งที่รัน (reproducibility)

---

### Step 7 — Modeling
เปรียบเทียบ 3 โมเดลพร้อมกัน:

```python
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}
```

| Model | Accuracy | Recall (Diabetes) | F1 (Diabetes) |
|-------|----------|-------------------|---------------|
| Logistic Regression | 0.79 | 0.63 | 0.64 |
| Decision Tree | 0.77 | 0.77 | 0.67 |
| **Random Forest** | **0.80** | **0.70** | **0.67** |

---

### Step 8 — Evaluation
วัดผลด้วย `accuracy_score` และ `classification_report`

**โมเดลที่เลือก: Random Forest**  
เพราะ balance ระหว่าง accuracy (0.80) และ recall (0.70) ดีที่สุด

> สำหรับงานตรวจโรค **recall สำคัญกว่า accuracy** — การพลาดคนเป็นเบาหวาน (False Negative) อันตรายกว่าการแจ้งเตือนผิด (False Positive)

**Confusion Matrix (Random Forest):**
```
                Predicted No  |  Predicted Yes
Actual No    [      86       |      16      ]  
Actual Yes   [      13       |      30      ]  ← 13 คนที่พลาด = False Negative
```

---

### Step 9 — Enhanced Visualization

**9A: Correlation Heatmap**  
แสดงความสัมพันธ์ระหว่างทุก feature พร้อมกันในภาพเดียว  
→ Glucose (0.49) แดงที่สุดในแถว Outcome ยืนยันว่าสำคัญที่สุด

**9B: Confusion Matrix เป็นภาพ**  
แสดง True/False Positive/Negative ด้วยสี ทำให้อ่านง่ายกว่าตัวเลข

---

### Step 10 — Cross Validation
แทนที่จะทดสอบครั้งเดียว ใช้ 5-fold cross validation เพื่อความน่าเชื่อถือ

```
Fold 1 → accuracy: 0.77
Fold 2 → accuracy: 0.72
Fold 3 → accuracy: 0.79
Fold 4 → accuracy: 0.82
Fold 5 → accuracy: 0.78
─────────────────────────
Mean: 0.78  Std: 0.03
```

**Std = 0.03** → โมเดล stable ไม่แกว่งมาก น่าเชื่อถือ

---

### Step 11 — Feature Importance
Random Forest บอกได้ว่าแต่ละ feature มีความสำคัญแค่ไหน

**ผลที่ได้ (เรียงจากมากไปน้อย):**
1. **Glucose** (~0.27) ← สำคัญที่สุด สอดคล้องกับ correlation
2. **BMI** (~0.17)
3. **Age** (~0.13)
4. DiabetesPedigreeFunction, Insulin, BloodPressure, SkinThickness, Pregnancies

**Experiment:** ทดลองใช้แค่ 4 features ที่สำคัญที่สุด  
→ Accuracy ลดจาก 0.78 → 0.76 แปลว่า features ที่เหลือยังมีประโยชน์อยู่

---

### Step 12 — GridSearchCV (Hyperparameter Tuning)
ลองทุก combination ของ hyperparameter อัตโนมัติ:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 10, None]
}
# ลอง 12 combinations × 5 folds = 60 รอบ
```

**Best params:** `max_depth=None, n_estimators=100`  
→ ค่า default ของ Random Forest ดีที่สุดอยู่แล้วสำหรับ dataset นี้

---

### Step 13 — sklearn Pipeline
รวม Scaler + Model ไว้ในขั้นตอนเดียว ป้องกัน data leakage และพร้อม deploy

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(max_depth=None, n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)   # scale + train ในบรรทัดเดียว
pipeline.predict(X_test)          # scale + predict ในบรรทัดเดียว
```

---

## Final Results

| Metric | Value |
|--------|-------|
| Model | Random Forest |
| Accuracy | 0.80 |
| Recall (Diabetes) | 0.70 |
| F1 (Diabetes) | 0.67 |
| Cross-val Mean | 0.78 ± 0.03 |

---

## Key Learnings

- Missing values ไม่ได้มีแค่รูปแบบ `NaN` — ต้องใช้ domain knowledge ตรวจจับ hidden zeros
- Correlation วัดทีละ feature แต่ ML รวมทุก feature พร้อมกัน จึงให้ผลต่างกัน
- Accuracy อย่างเดียวหลอกได้ — ต้องดู precision, recall, F1 ด้วยเสมอ
- การเลือกโมเดลขึ้นอยู่กับ use case ไม่ใช่แค่ตัวเลขสูงสุด

---

## Tech Stack

- Python 3.14
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

---

## How to Run

```bash
git clone https://github.com/<your-username>/pima-diabetes-classification
cd pima-diabetes-classification
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook pima_v2_practice.ipynb
```

---

## File Structure

```
pima-diabetes-classification/
├── pima_v2_practice.ipynb   # main notebook (13 steps)
├── diabetes.csv             # dataset
└── README.md                # this file
```
