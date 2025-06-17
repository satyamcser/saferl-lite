# 📦 Installation Guide

Follow these steps to install SafeRL-Lite locally or within a virtual environment.

---

## ✅ Requirements

- Python 3.8+
- `pip`
- `gymnasium`, `torch`, `shap`, `numpy`

---

## 🔧 Steps

### 1. Clone the Repository

```bash
git clone https://github.com/satyamcser/saferl-lite.git
cd saferl-lite
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Pre-Commit (Optional)

```bash
pre-commit install
pre-commit run --all-files
```

## Verify Installation

```bash
pytest tests/
```
