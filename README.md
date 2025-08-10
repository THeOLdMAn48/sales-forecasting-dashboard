# 📊 AI-Powered Sales Forecasting Dashboard

![Project Banner](banner.png)

## 🚀 Overview
An interactive **AI-powered Sales Forecasting Dashboard** that combines **Facebook Prophet** and **XGBoost** to predict future sales, generate automatic business insights, and visualize trends in an intuitive **Streamlit** interface.  
Designed for **real-world business use**, this tool helps decision-makers plan inventory, identify seasonal trends, and optimize sales strategies.

---

## 🛠️ Features
- **📈 Forecasting Models**: Facebook Prophet for time series forecasting & XGBoost for enhanced accuracy.
- **🔍 Automatic Insights**: Summarizes sales growth trends and peak months.
- **🎨 Interactive Visuals**: Beautiful and interactive plots with Matplotlib, Plotly, and Prophet components.
- **📥 Export Data**: Download forecasted results as CSV.
- **⚡ Easy Deployment**: Built with Streamlit for quick web deployment.

---

## 📂 Project Structure
```
sales-forecasting-dashboard/
│
├── data/
│ └── train.csv # Sample sales dataset
│
├── models/
│ ├── prophet_model.pkl # Saved Prophet model
│ ├── xgboost_model.pkl # Saved XGBoost model
│
├── notebooks/
│ └── exploration.ipynb # Data exploration & preprocessing
│
├── scripts/
│ ├── prophet_forecast.py # Prophet model forecasting
│ ├── xgboost_forecast.py # XGBoost model forecasting
│
├── app/
│ ├── streamlit_app.py # Main dashboard application
│ └── utils.py # Helper functions
│
├── banner.png # Project banner image
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── LICENSE
```
---

## 📊 Tech Stack
- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Prophet, XGBoost, Scikit-learn
- **App Framework:** Streamlit
- **Version Control:** Git, GitHub
---

## 📦 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/sales-forecasting-dashboard.git
cd sales-forecasting-dashboard
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the app
```bash
streamlit run app/streamlit_app.py
```
## 📷 Screenshots

