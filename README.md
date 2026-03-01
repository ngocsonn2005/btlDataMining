# btlDataMining
---

# 📊 Sentiment Analysis Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.24%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 1. Giới thiệu dự án

Dự án **Sentiment Analysis** được thực hiện trong khuôn khổ môn học **Data Mining**.

Mục tiêu chính của dự án là:

* Phân tích cảm xúc (Sentiment Analysis) từ dữ liệu đánh giá sản phẩm Amazon
* So sánh nhiều kỹ thuật khai phá dữ liệu khác nhau
* Xây dựng hệ thống trực quan hóa kết quả bằng Dashboard

Dự án áp dụng đầy đủ các hướng tiếp cận trong Data Mining:

* 🔗 **Association Rule Mining** – Khai phá luật kết hợp
* 🔍 **Clustering** – Phân cụm đánh giá
* 🤖 **Classification** – Phân lớp cảm xúc
* 🔄 **Semi-Supervised Learning** – Học bán giám sát

---

## 👥 Thông tin học phần

* **Môn học:** Data Mining
* **Giảng viên hướng dẫn:** ThS. Lê Thị Thùy Trang
* **Nhóm thực hiện:** Nhóm 7
* **Học kỳ:** II – Năm học 2025–2026

---

## 📁 2. Cấu trúc thư mục

```
SENTIMENT_ANALYSIS_PROJECT/
│
├── data/
│   ├── raw/                  # Dữ liệu gốc
│   └── processed/            # Dữ liệu đã xử lý
│
├── notebooks/                # Phân tích bằng Jupyter
│
├── src/                      # Source code chính
│
├── scripts/                  # Chạy pipeline tự động
│
├── outputs/                  # Kết quả (models, figures, tables)
│
├── app.py                    # Streamlit Dashboard
├── requirements.txt
├── README.md
└── .gitignore
```

Cấu trúc được thiết kế theo chuẩn **Machine Learning Project Structure**, dễ mở rộng và tái sử dụng.

---

## ⚙️ 3. Hướng dẫn cài đặt

### Bước 1: Clone repository

```bash
git clone https://github.com/ngocsonn2005/btlDataMining.git
cd btlDataMining
```

### Bước 2: Tạo môi trường ảo

**Sử dụng conda (khuyến nghị):**

```bash
conda create -n sentiment_env python=3.9
conda activate sentiment_env
```

Hoặc:

```bash
python -m venv sentiment_env
sentiment_env\Scripts\activate   # Windows
```

### Bước 3: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

## 📂 4. Chuẩn bị dữ liệu

Đặt file:

```
train.csv
test.csv
```

vào thư mục:

```
data/raw/
```

### Cấu trúc dữ liệu:

| Column      | Mô tả                      |
| ----------- | -------------------------- |
| label       | 1 = negative, 2 = positive |
| title       | Tiêu đề đánh giá           |
| review_text | Nội dung đánh giá          |

---

## ▶️ 5. Cách chạy dự án

### 🔹 Chạy toàn bộ pipeline

```bash
python scripts/run_pipeline.py
```

### 🔹 Chạy từng notebook

```bash
jupyter notebook hoặc chạy tất với lệnh python scripts/run_papermill.py
```

Mở và chạy lần lượt các notebook trong thư mục `notebooks/`.

### 🔹 Chạy Dashboard

```bash
streamlit run app.py
```

Truy cập tại:

```
http://localhost:8501
```

---

## 📊 6. Kết quả chính

### 🔗 Association Rules

* Tổng số luật: **2**
* Luật mạnh nhất: `book → read`
* Lift cao nhất: **2.36**
* Confidence: **0.55**

---

### 🔍 Clustering

* Số cụm tối ưu: **2**

  * **Cluster 0 (2.3%)** – Book / Fiction Reviews
  * **Cluster 1 (97.7%)** – Product / CD Reviews

---

### 🤖 Classification

| Metric    | Giá trị |
| --------- | ------- |
| Accuracy  | 0.8267  |
| Precision | 0.8298  |
| Recall    | 0.8306  |
| F1-Score  | 0.8302  |

🔹 **Mô hình tốt nhất:** Logistic Regression
🔹 Hiệu suất ổn định và thời gian huấn luyện nhanh

---

### 🔄 Semi-Supervised Learning

* Tỷ lệ dữ liệu có nhãn: 5%
* Số vòng self-training: 39
* Pseudo-labels thêm vào: 37,500
* Cải thiện tốt nhất: +0.0013

---

## 📈 7. Dashboard

Dashboard gồm 6 trang:

1. Overview
2. Association Rules
3. Clustering
4. Classification
5. Semi-Supervised
6. Reports

Tính năng:

* Biểu đồ tương tác (Plotly)
* So sánh mô hình
* Tải kết quả CSV
* Tự động phát hiện dữ liệu

---

## 🛠 8. Công nghệ sử dụng

* Python 3.9+
* Pandas, NumPy
* Scikit-learn
* TensorFlow / Keras
* XGBoost
* NLTK, Gensim
* MLxtend
* Streamlit
* Plotly
* Jupyter Notebook

---

## 🎯 9. Kết luận

Dự án đã:

✅ Ứng dụng đầy đủ các kỹ thuật khai phá dữ liệu
✅ So sánh nhiều mô hình khác nhau
✅ Xây dựng pipeline hoàn chỉnh
✅ Tạo dashboard trực quan hóa chuyên nghiệp

### Kết luận chính:

**Logistic Regression là mô hình phù hợp nhất cho bài toán Sentiment Analysis trên tập dữ liệu Amazon Reviews**, với:

* Hiệu suất cao
* Độ ổn định tốt
* Thời gian huấn luyện nhanh
* Dễ triển khai thực tế

---

## 📜 License

Dự án được phát hành theo giấy phép **MIT License**.

---

## 🙏 Lời cảm ơn

Xin chân thành cảm ơn **ThS. Lê Thị Thùy Trang** đã hướng dẫn tận tình trong suốt quá trình thực hiện dự án.

Cảm ơn các thành viên Nhóm 7 đã hợp tác và đóng góp để hoàn thành dự án.

---

## 📧 Liên hệ

* Nhóm 7 – Data Mining
* Email: [docongngocson2005@gmail.com](mailto:your-email@domain.com)

---

**© 2026 – Bài tập lớn | Data Mining | Học kỳ II – 2025–2026**

---
