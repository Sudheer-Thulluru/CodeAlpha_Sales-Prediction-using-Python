import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Load data (assuming the weird .xls extension is actually a CSV)
    df = pd.read_csv(r'C:\Headache\task-4\Advertising.csv.xls', index_col=0)

    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initial model evaluation
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2:  {r2_score(y_test, y_pred):.4f}")
    print(f"Intercept: {model.intercept_:.4f}\n")

    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    print("Coefficients:\n", coef_df.to_string(index=False), "\n")

    # Retrain on full dataset for scenario forecasting
    model.fit(X, y)

    avg_tv, avg_rad, avg_news = df['TV'].mean(), df['Radio'].mean(), df['Newspaper'].mean()

    # Build scenarios dynamically (cleaner than parallel lists)
    scenarios = [
        {'Scenario': 'Baseline', 'TV': avg_tv, 'Radio': avg_rad, 'Newspaper': avg_news},
        {'Scenario': 'Heavy TV', 'TV': 250.0, 'Radio': 15.0, 'Newspaper': 10.0},
        {'Scenario': 'Radio Focus', 'TV': 50.0, 'Radio': 75.0, 'Newspaper': 0.0},
        {'Scenario': 'Equal Spread', 'TV': 100.0, 'Radio': 100.0, 'Newspaper': 100.0},
        {'Scenario': 'News to Radio', 'TV': avg_tv, 'Radio': avg_rad + avg_news, 'Newspaper': 0.0}
    ]
    
    scenarios_df = pd.DataFrame(scenarios)

    # Generate predictions and calculate budgets
    scenarios_df['Predicted_Sales'] = model.predict(scenarios_df[['TV', 'Radio', 'Newspaper']]).round(2)
    scenarios_df['Total_Budget'] = scenarios_df[['TV', 'Radio', 'Newspaper']].sum(axis=1).round(2)

    # Reorder columns for display
    cols = ['Scenario', 'Total_Budget', 'TV', 'Radio', 'Newspaper', 'Predicted_Sales']
    print("Forecasts:\n", scenarios_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()