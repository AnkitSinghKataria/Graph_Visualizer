# 📊 GraphVisualizer

GraphVisualizer is a beginner-friendly Python library to simplify **Exploratory Data Analysis (EDA)** using visualizations. It supports **Seaborn**, **Matplotlib**, and **Plotly**, and guides users interactively through different analysis types like univariate, bivariate, multivariate, and time-series visualizations.

---

## ✨ Features

- 📚 Simple interactive menu-driven system (ideal for Jupyter notebooks)
- 🔍 Univariate, Bivariate, Multivariate & Datetime Analysis
- 🎨 Choose between Seaborn, Matplotlib, and Plotly
- 🧠 Automatic hue/color support
- 📸 Save plots as images
- 🧾 Add plots to dashboard & export all together
- 📅 Time series with daily/monthly/yearly resampling
- 📦 Designed to be extended for beginners, teachers & analysts

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/your-username/graph-visualizer.git
cd graph-visualizer
pip install -r requirements.txt

from graph_visualizer import GraphVisualizer
import seaborn as sns

# Load your DataFrame
df = sns.load_dataset("tips")  # or your own CSV file

# Launch the visualizer
gv = GraphVisualizer(df)
gv.run()


graph-visualizer/
│
├── graph_visualizer.py       # Main class
├── examples/
│   └── example_notebook.ipynb
├── README.md
├── LICENSE
├── .gitignore
└── requirements.txt
