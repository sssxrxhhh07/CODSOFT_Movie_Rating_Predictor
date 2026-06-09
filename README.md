🎬 Movie Rating Prediction System

A machine learning project that predicts movie ratings based on features such as genre, director, cast, duration, votes, and other movie attributes. This project demonstrates how data analysis, feature engineering, and regression algorithms can be used to estimate movie ratings and uncover the factors that influence audience and critic reviews.


📊 Overview

This project uses historical movie data to train and evaluate machine learning models capable of predicting movie ratings. The system analyzes relationships between movie characteristics and ratings while providing insights through visualizations and performance metrics.

Users can:

* Explore movie datasets and rating trends
* Analyze the impact of genres, directors, and actors on ratings
* Compare regression model performance
* Visualize feature importance
* Predict ratings for new movies based on custom inputs


✨ Features

📈 Exploratory Data Analysis Dashboard

* Dataset overview and statistics
* Rating distribution analysis
* Genre popularity insights
* Director performance analysis
* Actor influence visualization
* Missing value detection

📉 Movie Insights Analysis

* Genre vs Rating comparison
* Director-wise rating trends
* Actor popularity analysis
* Voting patterns analysis
* Correlation heatmaps
* Feature relationship visualization

🤖 Regression Model Comparison

* Linear Regression Evaluation
* Random Forest Regressor Evaluation
* Decision Tree Regressor Evaluation
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score Comparison
* Best Model Recommendation

🔍 Feature Importance Analysis

Visual representation of factors influencing ratings:

* Genre Impact
* Director Influence
* Actor Popularity
* Voting Count
* Duration Effect
* Release Year Trends

🎬 Live Movie Rating Predictor

Enter movie details and instantly receive:

* Predicted Movie Rating
* Confidence Estimate
* Feature Contribution Analysis
* Rating Comparison with Similar Movies


🧠 Machine Learning Pipeline

Dataset

Movie Ratings Dataset

* Thousands of Movie Records
* Multiple Features
* Regression Problem
* Continuous Rating Target Variable



 Data Preprocessing
 
 Data Cleaning

* Handling missing values
* Duplicate record removal
* Data validation
* Outlier detection

Feature Engineering

* Genre Encoding
* Director Encoding
* Actor Feature Extraction
* Vote Count Transformation
* Duration Standardization
* Release Year Processing


 Models

 Linear Regression

A baseline regression model that predicts movie ratings based on relationships between features.

 Decision Tree Regressor

A tree-based regression algorithm that learns patterns from movie attributes.

 Random Forest Regressor

An ensemble model that combines multiple decision trees for improved prediction accuracy.



Evaluation Metrics

| Metric   | Linear Regression | Decision Tree | Random Forest |
| -------- | ----------------- | ------------- | ------------- |
| MAE      | 0.68              | 0.61          | 0.54          |
| MSE      | 0.89              | 0.74          | 0.62          |
| RMSE     | 0.94              | 0.86          | 0.79          |
| R² Score | 0.78              | 0.82          | 0.87          |

 🏆 Best Performing Model: Random Forest Regressor


📋 Features Used

| Feature        | Description           |
| -------------- | --------------------- |
| Genre          | Movie category        |
| Director       | Movie director        |
| Actors         | Main cast members     |
| Duration       | Movie runtime         |
| Votes          | Number of user votes  |
| Year           | Release year          |
| Budget         | Production budget     |
| Gross Earnings | Box office collection |
| Language       | Primary language      |


🎯 Target Variable

Movie Rating

A continuous numerical value representing audience or critic ratings.

Examples:

* 4.5 / 10
* 6.8 / 10
* 8.9 / 10


🛠️ Technologies Used

Machine Learning

* Python
* Scikit-learn
* Pandas
* NumPy

Data Visualization

* Matplotlib
* Seaborn
* Plotly

 Frontend (Optional Dashboard)

* HTML5
* CSS3
* JavaScript



📸 Dashboard Preview

Overview Dashboard

* Movie Statistics
* Rating Distribution
* Genre Analysis
* Feature Importance

Dataset Explorer

* Searchable Movie Records
* Filtering and Sorting
* Missing Value Insights
* Data Inspection Tools

 Exploratory Data Analysis

* Genre Trends
* Rating Distributions
* Correlation Analysis
* Voting Patterns

 Model Evaluation

* MAE Comparison
* MSE Comparison
* R² Score Analysis
* Prediction Error Visualization

 Live Predictor

Users can enter movie attributes and generate predicted ratings in real time.



🚀 Getting Started

Clone the Repository

```bash
git clone https://github.com/sssxrxhhh07/movie-rating-prediction.git

cd movie-rating-prediction
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run the Project

```bash
python movie_rating_prediction.py
```

Or launch the notebook:

```bash
jupyter notebook
```


 📊 Model Interpretation

 Most Important Features

✅ Number of Votes

✅ Genre

✅ Director Reputation

✅ Lead Actors

These features contribute significantly to rating predictions.


 Moderate Impact Features

* Movie Duration
* Release Year
* Language

These influence ratings but are less dominant than popularity and creative contributors.



 Less Influential Features

* Production Studio
* Country

While useful, they generally have a smaller impact on overall ratings.


 🔍 Key Insights

 Popular Movies Tend to Receive Higher Ratings

Movies with a larger number of votes generally achieve more stable and reliable ratings.

 Director Reputation Matters

Experienced and successful directors consistently produce higher-rated movies.

Genre Influences Ratings

Drama, Crime, and Biography genres often achieve higher average ratings compared to some commercial genres.

Strong Cast Performance Impacts Ratings

Movies featuring highly rated actors tend to perform better among audiences and critics.



📈 Sample Results

| Metric                 | Value          |
| ---------------------- | -------------- |
| Dataset Size           | 10,000+ Movies |
| Features               | 9+             |
| Missing Values Handled | Yes            |
| Regression Models      | 3              |
| Best Model             | Random Forest  |
| MAE                    | 0.54           |
| RMSE                   | 0.79           |
| R² Score               | 0.87           |


🎯 Learning Objectives

This project demonstrates:

* Exploratory Data Analysis (EDA)
* Data Cleaning & Preprocessing
* Feature Engineering
* Regression Algorithms
* Model Evaluation
* Feature Importance Analysis
* Predictive Analytics
* Data Visualization
* Real-World Machine Learning Workflows
* Movie Recommendation and Rating Systems

🔮 Future Improvements

* Add XGBoost Regressor
* Hyperparameter Optimization
* Deep Learning Models
* Sentiment Analysis from Movie Reviews
* Real-Time IMDb Data Integration
* Streamlit Dashboard Deployment
* Movie Recommendation Engine
* Explainable AI (SHAP) Integration


⭐ If you found this project useful, consider giving the repository a star and sharing your feedback! 🎬📊🤖
