import dash
from dash import dcc
from dash import html
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats import diagnostic
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
import plotly.express as px
import dash_table
import numpy as np
import scipy.stats as stats
import base64
import io
from tabulate import tabulate

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Multiple Variable Time-Series Regression and Causality Analysis in Python using Dash and Plotly"),
    dcc.Upload(
        id="upload-data",
        children=html.Div([
            "Drag and Drop or ",
            html.A("Select Files")
        ]),
        style={
            "width": "50%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px"
        },
        multiple=False
    ),
    html.Div(id="output-div"),
    dcc.Graph(id="result-plot"),
    html.Div(id="tests-summary-tables"),
    html.Div(id="regression-summary-table")
])

# Define the callback function for file upload, regression, and tests
@app.callback(
    [dash.dependencies.Output("output-div", "children"),
     dash.dependencies.Output("result-plot", "figure"),
     dash.dependencies.Output("tests-summary-tables", "children"),
     dash.dependencies.Output("regression-summary-table", "children")],
    [dash.dependencies.Input("upload-data", "contents"),
     dash.dependencies.Input("upload-data", "filename")]
)
def perform_regression(contents, filename):
    # Check if a file has been uploaded
    if contents is not None:
        # Read the uploaded file as a DataFrame
        content_type, content_string = contents.split(",")
        decoded_content = base64.b64decode(content_string)
        try:
            if "csv" in filename:
                # Assuming the first row as headers and rest as Y values
                data = pd.read_csv(io.StringIO(decoded_content.decode("utf-8")))
            else:
                return "Invalid file format. Please upload a CSV file.", {}, "", ""
        except Exception as e:
            return f"Error occurred while reading the file: {str(e)}", {}, "", ""

        # Get the column names from the DataFrame
        column_names = data.columns.tolist()

        # Extract date/year, X, and Y column names
        date_column = column_names[0]
        x_column = column_names[1]
        y_columns = column_names[2:]

        # Convert the date/year column to datetime type
        data[date_column] = pd.to_datetime(data[date_column])

        # Add a constant column to the X variables for the intercept term
        X = sm.tools.add_constant(data[x_column])

        # Perform linear regression for each Y column
        results = []
        for y_column in y_columns:
            model = sm.OLS(data[y_column], X)
            result = model.fit()
            results.append(result)

        # Create the regression summary table
        summary_tables = [result.summary2().tables[1] for result in results]
        summary_df = pd.concat(summary_tables)

        



        # Create the scatter plot of the data points and the regression lines
        fig = px.scatter(data_frame=data, x=date_column, y=y_columns, title="Multiple Variable Linear Regression")
        for result, y_column in zip(results, y_columns):
            fig.add_scatter(x=data[date_column], y=result.fittedvalues, mode="lines", name=f"Regression Line ({y_column})")

        # Perform tests
        tests_output = []
        for result, y_column in zip(results, y_columns):
            # Test 1: Linear relationship between slope and intercept
            slope, intercept = result.params[1], result.params[0]
            tests_output.append(f"Test 1: Linear relationship between slope and intercept ({y_column})")
            tests_output.append(f"Slope: {slope:.4f}")
            tests_output.append(f"Intercept: {intercept:.4f}")
            

            # Test 2: Independent variable is not random
            x = data[x_column]
            residuals = result.resid
            pearson_r, pearson_p = stats.pearsonr(x, residuals)
            tests_output.append("")
            tests_output.append(f"Test 2: Pearson correlation coefficient between X and residuals ({y_column})")
            tests_output.append(f"Pearson R: {pearson_r:.4f}")
            tests_output.append(f"P-value: {pearson_p:.4f}")
            tests_output.append("")
            

            # Test 3: Residuals (errors) are zero
            mean_residual = np.mean(residuals)
            tests_output.append(f"Test 3: Mean of residuals ({y_column})")
            tests_output.append(f"Mean: {mean_residual:.4f}")
            

            # Test 4: Residuals (errors) are constant
            _, jb_p = stats.jarque_bera(residuals)
            tests_output.append(f"Test 4: Jarque-Bera test for residuals ({y_column})")
            tests_output.append(f"P-value: {jb_p:.4f}")
            

            # Test 5: Residuals (errors) are not correlated
            _, ljung_box_p = diagnostic.acorr_ljungbox(residuals, lags=[1])
            tests_output.append(f"Test 5: Ljung-Box test for residuals ({y_column})")
            if isinstance(ljung_box_p[0], str):
                tests_output.append(f"P-value: {ljung_box_p[0]}")
            else:
                tests_output.append(f"P-value: {float(ljung_box_p[0]):.4f}")
            

            # Test 6: Residuals (errors) follow a normal distribution
            _, normality_p = stats.normaltest(residuals)
            tests_output.append(f"Test 6: Normality test for residuals ({y_column})")
            tests_output.append(f"P-value: {normality_p:.4f}")
            

            # Test 7: Stationary test
            adf_result = adfuller(residuals)
            tests_output.append(f"Test 7: Stationary test for residuals ({y_column})")
            tests_output.append(f"ADF Statistic: {adf_result[0]:.4f}")
            tests_output.append(f"P-value: {adf_result[1]:.4f}")
            tests_output.append("")

            # Check if residuals are stationary
            if adf_result[1] > 0.05:
                tests_output.append("Residuals are not stationary. Proceeding with differencing...")

                # Differencing until residuals become stationary
                stationary_res = residuals.diff().dropna()
                differencing_steps = 1
                while True:
                    adf_result = adfuller(stationary_res)
                    tests_output.append(f"Differencing Step {differencing_steps}")
                    tests_output.append(f"ADF Statistic: {adf_result[0]:.4f}")
                    tests_output.append(f"P-value: {adf_result[1]:.4f}")
                    tests_output.append("")

                        

                    if adf_result[1] <= 0.05:
                        tests_output.append(f"Residuals became stationary after {differencing_steps} differencing steps.")
                        residuals = stationary_res
                        break

                    stationary_res = stationary_res.diff().dropna()
                    differencing_steps += 1
            else:
                tests_output.append("Residuals are already stationary.")
                tests_output.append("")

            # Test 8: Toda-Yamamoto Granger causality test
            maxlag = int(np.ceil(12 * np.power(len(data) / 100.0, 1 / 4)))
            granger_result = grangercausalitytests(data[[y_column, x_column]], maxlag=maxlag, verbose=False)
            tests_output.append(f"Test 8: Toda-Yamamoto Granger causality test ({y_column} -> {x_column})")
            tests_output.append("")
            for lag in range(1, maxlag + 1):
                tests_output.append(f"Lag {lag}")
                tests_output.append(f"F-value: {granger_result[lag][0]['ssr_ftest'][0]:.4f}")
                tests_output.append(f"P-value: {granger_result[lag][0]['ssr_ftest'][1]:.4f}")
                tests_output.append("")
            

        # Create separate tables for each test result
        test_tables = []
        for i in range(0, len(tests_output), 2):
            if i + 1 < len(tests_output):  # Check if index is within range
                test_name = tests_output[i]
                test_result = tests_output[i + 1]
                test_table = pd.DataFrame({"Test": [test_name], "Output": [test_result]})
                test_table = dash_table.DataTable(
                    columns=[{"name": col, "id": col} for col in test_table.columns],
                    data=test_table.to_dict("records"),
                    style_cell={"textAlign": "left"},
                    style_header={"fontWeight": "bold"},
                )
                test_tables.append(test_table)
                test_tables.append(html.Hr())


        # Create the regression summary table main
        regression_summaries = []
        for result, y_column in zip(results, y_columns):
            summary_table_2 = result.summary().tables[0]
            summary_df_2 = pd.DataFrame(summary_table_2.data[1:], columns=summary_table_2.data[0])
            summary_df_2.columns = [f"{col} ({y_column})" for col in summary_df_2.columns]
            regression_summaries.append(html.Div(f"Regression Summary ({y_column})"))
            regression_summaries.append(dash_table.DataTable(
                columns=[{"name": col, "id": col} for col in summary_df_2.columns],
                data=summary_df_2.to_dict("records"),
                style_cell={"textAlign": "left"},
                style_header={"fontWeight": "bold"},
            ))


        return [
            dash_table.DataTable(
                columns=[{"name": col, "id": col} for col in summary_df.columns],
                data=summary_df.to_dict("records"),
                style_cell={"textAlign": "left"},
                style_header={"fontWeight": "bold"},
            ),
            fig,
            html.Div(regression_summaries),
            html.Div(test_tables)
            
        ]
    


    # If no file has been uploaded yet, return empty values
    return "", {}, "", ""


if __name__ == "__main__":
    app.run_server(debug=True)
