# 🏥 Emergency Room Wait Time Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://manishkondoju-er-wait-time-predictor.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

## 🔗 Live Demo

**[Try the App Here →](https://manishkondoju-er-wait-time-predictor.streamlit.app)**

## 📋 Overview

An intelligent machine learning application that predicts emergency room wait times based on various factors including patient symptoms, triage level, time of arrival, and current hospital occupancy. Built to help patients make informed decisions about when to visit the ER and what to expect.

### 🎯 Key Features

- **Real-time Predictions**: Instant wait time estimates based on current conditions
- **High Accuracy**: Achieves ~85% prediction accuracy using Random Forest algorithm
- **Interactive Dashboard**: User-friendly interface with visual analytics
- **Temporal Patterns**: Accounts for time-of-day, day-of-week, and seasonal variations
- **Smart Recommendations**: Provides actionable advice based on predicted wait times

## 🤖 Model Performance

| Metric | Score |
|--------|--------|
| **R² Score** | 0.856 |
| **Mean Absolute Error** | ~28 minutes |
| **Training Samples** | 5,000 |
| **Features** | 25+ engineered features |
| **Algorithm** | Random Forest (100 estimators) |

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.11+** - Primary programming language
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning library
- **XGBoost** - Advanced gradient boosting

### Data & Visualization
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Plotly** - Interactive visualizations

### Deployment
- **Streamlit Cloud** - Cloud hosting platform
- **GitHub** - Version control and CI/CD

## 📊 Features in Detail

### Patient Input Parameters
- **Chief Complaint** (8 categories including Chest Pain, Fracture, Fever, etc.)
- **Age** (1-120 years)
- **Triage Level** (1-5, where 1 is most critical)
- **Arrival Mode** (Walk-in vs Ambulance)
- **Current ER Occupancy** (0-100%)
- **Time of Arrival** (with automatic current time option)

### Prediction Outputs
- Estimated wait time in minutes and hours
- Color-coded status indicator (Green/Yellow/Red)
- Interactive gauge visualization
- Personalized recommendations
- Historical wait time patterns

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Virtual environment (recommended)

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/ManishKondoju/er-wait-time-predictor.git
cd er-wait-time-predictor
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model**
```bash
python train.py
```

5. **Run the application**
```bash
streamlit run app/streamlit_app.py
```

6. **Access the app**
```
Open browser to http://localhost:8501
```

## 📁 Project Structure

```
er-wait-time-predictor/
│
├── app/
│   └── streamlit_app.py       # Main Streamlit application
│
├── src/
│   └── data_generator.py      # Synthetic data generation
│
├── models/
│   ├── model.pkl              # Trained Random Forest model
│   ├── scaler.pkl             # Feature scaler
│   └── feature_cols.pkl      # Feature column names
│
├── data/
│   └── raw/
│       └── er_visits.csv     # Generated training data
│
├── train.py                   # Model training script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                # Git ignore rules
```

## 🔬 How It Works

### Data Generation
The system uses synthetic data that simulates realistic ER visit patterns:
- Temporal variations (hourly, daily, seasonal)
- Complaint-specific base wait times
- Occupancy-based adjustments
- Triage priority factors

### Feature Engineering
- **Temporal Features**: Hour, day of week, month with cyclical encoding
- **Binary Indicators**: Weekend, night shift, evening rush
- **Categorical Encoding**: One-hot encoding for complaints and arrival mode
- **Interaction Effects**: Combined features for complex patterns

### Model Pipeline
1. Data generation with realistic patterns
2. Feature engineering and preprocessing
3. StandardScaler normalization
4. Random Forest training with cross-validation
5. Model serialization for deployment

## 📈 Future Enhancements

- [ ] Integration with real hospital APIs
- [ ] Multi-hospital support and comparison
- [ ] Mobile responsive design improvements
- [ ] Historical wait time tracking
- [ ] SMS/Email notifications for wait time updates
- [ ] Insurance provider integration
- [ ] Multilingual support
- [ ] Advanced deep learning models (LSTM for time series)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Achievements

- Successfully deployed ML model to production
- 85% prediction accuracy on test data
- Real-time predictions with <100ms latency
- Interactive visualizations for better UX

## 👤 Author

**Manish Kondoju**

- GitHub: [@ManishKondoju](https://github.com/ManishKondoju)
- LinkedIn: [Connect with me](https://linkedin.com/in/your-profile)
- Portfolio: [View More Projects](https://your-portfolio-site.com)

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- Scikit-learn community for ML tools
- Healthcare professionals for domain insights
- Open source contributors

## 📞 Contact

For questions or feedback about this project:
- Open an issue on GitHub
- Email: your.email@example.com

## ⚠️ Disclaimer

This is a predictive model created for educational and demonstration purposes. Actual emergency room wait times may vary significantly. In case of a medical emergency, always call 911 or your local emergency services immediately. This tool should not be used as a substitute for professional medical advice.

---

<div align="center">
  Made with ❤️ using Python and Streamlit
  <br>
  <strong>If you found this helpful, please ⭐ this repository!</strong>
</div>
