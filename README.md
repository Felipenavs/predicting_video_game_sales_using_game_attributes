# Predicting Video Game Sales Using Game Attributes

A Data Science project that predicts global video game sales from attributes such as genre, platform, publisher, ESRB rating, and review scores.

**Authors:** Felipe Mancia Navas · Youngseok Lee ·  Joseph Dasti 

**Course:** Intro to Data Science 439

---

## Dataset

This project uses the **Video Game Sales with Ratings** dataset compiled by Rush Kirubi on Kaggle.

1. Go to: https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings
2. Sign in with a free Kaggle account
3. Click **Download**
4. create a `data/` folder in the project root.
5. Extract the downloaded zip file into the `data/` folder
6. Rename it to exactly: `Video_Games_Sales_as_at_22_Dec_2016.csv`

---

## Setup

Make sure you have **Python 3.8+** installed, then:

```bash
# 1. Clone the repo
git clone https://github.com/your-username/predicting_video_game_sales_using_game_attributes.git
cd predicting_video_game_sales_using_game_attributes

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```
---

## Running the Code

Run the files in this order from the project root:

**1. Exploratory data analysis**

Generates EDA plots in `figures/`

```bash
python eda.py
```
**2. Preprocessing**

Cleans the dataset and creates the train/test files in data/

```bash
python preprocessing.py
```

**Model training and evaluation**

Trains the regression models, saves evaluation figures in figures/, and saves model files in models/

```bash
python train.py
```
---

## Requirements

All dependencies are listed in `requirements.txt`. Key libraries:

- `pandas` — data loading and preprocessing
- `numpy` — numerical operations
- `matplotlib` / `seaborn` — visualizations
- `scikit-learn` — regression models and evaluation

