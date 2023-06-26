# TSRegressionToolkit
An all-in-one toolkit for time-series regression analysis in Python. Perform regression tests, visualize relationships, and gain deeper insights into your time-dependent data with ease.
# TimeSeriesRegressor
The TimeSeriesRegressor is a Python library for time-series regression analysis and forecasting. It provides comprehensive statistical tests and visualizations to help you unlock insights and uncover patterns in your time-dependent data.

# Regression Analysis – Linear Model Assumptions
 Linear regression analysis is based on six fundamental assumptions:

 ### The dependent and independent variables show a linear relationship between the slope and the intercept.
   1. The independent variable is not random.
   2. The value of the residual (error) is zero.
   3. The value of the residual (error) is constant across all observations.
   4. The value of the residual (error) is not correlated across all observations.
   5. The residual (error) values follow the normal distribution.

# Features
 1. Perform multiple variable time-series regression analysis
 2. Conduct statistical tests for assessing relationships and causality between variables
 3. Generate visualizations such as line charts, scatter plots, and heatmaps to aid in data analysis
 4. Handle CSV file uploads and process data for regression analysis
 5. Support for dashboards and interactive data visualization using Dash and Plotly

# Tests in the code
  Test 1: Linear Relationship between Slope and Intercept
  This test examines the linear relationship between the slope and intercept of the regression line for each Y variable. We calculate the slope and intercept values and analyze their significance.
  
  Test 2: Pearson Correlation Coefficient
  We assess the correlation between the X variable and the residuals (errors) of the regression model for each Y variable. The Pearson correlation coefficient and its p-value provide insights into the relationship between the independent variable and the model's residuals.
  
  Test 3: Mean of Residuals
  We calculate the mean of the residuals for each Y variable to determine if they are close to zero. A non-zero mean may indicate the presence of systematic errors in the regression model.
  
  Test 4: Jarque-Bera Test for Residuals
  The Jarque-Bera test assesses the normality of the residuals by examining skewness and kurtosis. We analyze the p-value associated with the test to determine if the residuals follow a normal distribution.
  
  Test 5: Ljung-Box Test for Residuals
  The Ljung-Box test checks the autocorrelation of the residuals at lag 1. By examining the p-value of the test, we can determine if the residuals are correlated or exhibit randomness.
  
  Test 6: Normality Test for Residuals
  We perform a normality test on the residuals using a statistical test such as the Shapiro-Wilk or Kolmogorov-Smirnov test. This helps us assess if the residuals are normally distributed.
  
  Test 7: Stationarity Test for Residuals
  To determine if the residuals are stationary, we conduct the Augmented Dickey-Fuller (ADF) test. This test helps us evaluate if the residuals exhibit a constant mean and variance over time. The p-value from the ADF test provides insights into the stationarity of the residuals.
  
  Causality Analysis
  In addition to regression analysis, we can also explore causality between variables. Causality analysis helps us understand the directional relationship between variables and determine if changes in one variable have an impact on another. Here are some causality analysis techniques we use:
  
  Granger Causality Test
  The Granger causality test examines if one variable can predict another variable by utilizing the concept of time lag. We apply this test to assess if the X variable has a causal relationship with each Y variable. The results provide the Granger causality statistics and associated p-values.
  
  Pairwise Causality Heatmap
  To visualize the causality relationships between variables, we create a pairwise causality heatmap. This heatmap displays the strength and directionality of the causal links between the X variable and each Y variable. It helps identify the variables that have a significant influence on others.
  
  Visualizing Results
  To enhance the understanding of the analysis, we employ interactive visualizations using the Dash and Plotly libraries. These visualizations allow for dynamic exploration of the data and results. We can create line charts to display the time series of variables, scatter plots to show the relationships between variables, and heatmaps to visualize causality.
# Installation
  To use the TimeSeriesRegressor library, you need to have Python 3.x installed. You can install the library and its dependencies using pip:
    
    pip install timeseriesregressor

# Usage
  Import the necessary modules:

    import timeseriesregressor as tsr
    import pandas as pd
  Read the data from a CSV file:
    
    data = pd.read_csv('New_data.csv')
  Perform time-series regression analysis:
    
    regressor = tsr.TimeSeriesRegressor()
    regressor.fit(data, target_column='target', feature_columns=['feature1', 'feature2'])
  Visualize the results:
    
    regressor.plot_results()
  Interpret the statistical tests:
     
     regressor.run_tests()
For more detailed usage instructions and examples, please refer to the documentation.

Contributing
We welcome contributions from the community to enhance the functionality and usability of the TimeSeriesRegressor library. If you find any bugs, have feature requests, or would like to contribute code, please submit an issue or a pull request on the GitHub repository.
