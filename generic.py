import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge, HuberRegressor
from sklearn.tree import DecisionTreeRegressor

# Set the theme configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define custom CSS styles
custom_styles = """
<style>
body {
    background-color: #f2f2f2;
    color: #333;
    font-family: Arial, sans-serif;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: #fff;
    border-radius: 10px;
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
}



.sidebar .stAccordion {
    background-color: #fff;
    border-radius: 5px;
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
}

.stButton {
    background-color: #007bff;
    color: #fff;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
}

.stButton:hover {
    background-color: #0056b3;
}

.dataframe {
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 14px;
}

.dataframe th, .dataframe td {
    padding: 8px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

.dataframe th {
    background-color: #f2f2f2;
}

.subheader {
    font-size: 24px;
    margin-top: 30px;
}



/* Add more custom styles here */

</style>
"""

def main():
    st.markdown(custom_styles,
            unsafe_allow_html=True
        )
     # Display the main title with custom styles
    st.markdown("<h2 style='text-align:center; color:#007BFF;'>Car Price Prediction Model</h2>", unsafe_allow_html=True)

    
    st.write("Upload an Excel or CSV file")

    # Upload the Excel file using Streamlit file uploader
    uploaded_file = st.file_uploader("Upload your file:", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]

        if file_extension.lower() == "csv":
            # Read the CSV file using pandas
            car = pd.read_csv(uploaded_file)

            # Convert DataFrame to XLSX
            converted_file = io.BytesIO()
            car.to_excel(converted_file, index=False)
            converted_file.seek(0)

        # Read the Excel file into a pandas DataFrame
        else:
            converted_file = io.BytesIO(uploaded_file.read())
            car = pd.read_excel(converted_file)

        # Extract the CompanyName from CarName column
        Company_Name = car['CarName'].apply(lambda x: x.split(' ')[0])

        # Insert the CompanyName column
        car.insert(3, "CompanyName", Company_Name)

        # Drop the CarName column
        car.drop(['CarName'], axis=1, inplace=True)

        # Apply label encoding to categorical columns
        X = car.apply(lambda col: LabelEncoder().fit_transform(col))
        X = X.drop(['CompanyName', 'price'], axis=1)
        y = car['price']

        # Apply PCA
        pca = PCA(n_components=0.99)
        x_reduced = pca.fit_transform(X)

        # Split the data into training and testing sets
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(x_reduced, y, test_size=0.2, random_state=42)

        # Dictionary to store evaluation results
        clean_evals = dict()
        reduced_evals = dict()

        def evaluate_regression(evals, model, name, X_train, X_test, y_train, y_test):
            train_error = mean_squared_error(y_train, model.predict(X_train), squared=False)
            test_error = mean_squared_error(y_test, model.predict(X_test), squared=False)
            r2_train = r2_score(y_train, model.predict(X_train))
            r2_test = r2_score(y_test, model.predict(X_test))
            evals[str(name)] = [train_error, test_error, r2_train, r2_test]
            print("Training Error " + str(name) + " {}  Test error ".format(train_error) + str(name) + " {}".format(test_error))
            print("R2 score for " + str(name) + " training is {} ".format(r2_train * 100) + " and for test is {}".format(
                r2_test * 100))

        # Reduced Data Linear Regression
        reduced_lr = LinearRegression().fit(X_test_r, y_test_r)
        evaluate_regression(reduced_evals, reduced_lr, "Linear Regression", X_train_r, X_test_r, y_train_r, y_test_r)

        # Reduced Lasso Regression
        reduced_las = Lasso().fit(X_test_r, y_test_r)
        evaluate_regression(reduced_evals, reduced_las, "Lasso Regression", X_train_r, X_test_r, y_train_r, y_test_r)

        # Reduced Ridge Regression
        reduced_rlr = Ridge().fit(X_test_r, y_test_r)
        evaluate_regression(reduced_evals, reduced_rlr, "Ridge Regression", X_train_r, X_test_r, y_train_r, y_test_r)

        # Reduced Robust Regression
        huber_r = HuberRegressor().fit(X_test_r, y_test_r)
        evaluate_regression(reduced_evals, huber_r, "Huber Regression", X_train_r, X_test_r, y_train_r, y_test_r)

        # Reduced Decision Tree Regression
        dt_r = DecisionTreeRegressor(max_depth=5, min_samples_split=10).fit(X_test_r, y_test_r)
        evaluate_regression(reduced_evals, dt_r, "Decision Tree Regression", X_train_r, X_test_r, y_train_r, y_test_r)

        # Reduced Random Forest Regression
        ##rf_r = RandomForestRegressor(n_estimators=15).fit(X_test_r, y_test_r)
        rf_r = RandomForestRegressor(n_estimators=15, random_state=42).fit(X_test_r, y_test_r)
        evaluate_regression(reduced_evals, rf_r, "Random Forest Regression", X_train_r, X_test_r, y_train_r, y_test_r)

        # Create a DataFrame for evaluation results
        eval_df = pd.DataFrame.from_dict(reduced_evals, orient='index',
                                         columns=['Train Error', 'Test Error', 'R2 Score (Train)', 'R2 Score (Test)'])

        # Display the modified DataFrame
        st.write("#### Cleaned Data:")
        st.write('**Data:** ' + str(car.shape[0]) + ' rows and ' + str(car.shape[1]) + ' columns.')
        st.dataframe(car)

        st.subheader("Regression Model Evaluation Results:")
        if 'R2 Score (Test)' in eval_df.columns:
            # Set the desired height and width using CSS style
            eval_df_html = eval_df['R2 Score (Test)'].to_frame().to_html()
            eval_df_html = f'<div style="height: 300px; width: 500px; overflow: auto;">{eval_df_html}</div>'
            st.markdown(eval_df_html, unsafe_allow_html=True)
        else:
            st.write("Evaluation results for some models are missing.")

        
        # Display the modified DataFrame
        st.write("#### Visualising features:")

        def display_option_data(selected_option):
            # Replace with your own logic to retrieve and display the data for the selected option
            if selected_option == 'Intercorrelation':
                # Use st.checkbox to display checkboxes in a single row
                col1, col2 = st.columns(2)
                Matrix = col1.checkbox("Correlation Matrix")
                Heatmap = col2.checkbox("Heatmap")

                
                # Display the correlation matrix
                if Matrix:
                    numerical_cols = car.select_dtypes(include=[np.number]).columns
                    correlation_matrix = car[numerical_cols].corr()
                    st.write("##### Correlation Matrix :")
                    st.write('**Correlation Data:** ' + str(correlation_matrix.shape[0]) + ' rows and ' + str(correlation_matrix.shape[1]) + ' columns.')
                    st.dataframe(correlation_matrix)
                
                # Displaying the Heatmap
                if Heatmap:
                    numeric_columns = car.select_dtypes(include=[np.number])
                    numeric_columns = numeric_columns.drop(columns=['car_ID', 'symboling'])
                    corr = numeric_columns.corr()
                    # Display the correlation matrix as an interactive heatmap
                    st.write("##### Intercorrelation Matrix Heatmap:")
                    fig = go.Figure(data=go.Heatmap(z=corr.values,
                                                    x=numeric_columns.columns,
                                                    y=numeric_columns.columns))
                    fig.update_layout(width=700, height=500)  # Set the size of the heatmap
                    st.plotly_chart(fig)

              
            if selected_option == 'Price Vs. Feature':
                exclude_columns = ['symboling', 'car_ID', 'price','CompanyName']  # Replace with the actual column names you want to exclude
                column_names = [col for col in car.columns.tolist() if col not in exclude_columns]

                # Create a Streamlit multiselect dropdown and populate it with the column names
                selected_columns = st.multiselect('Select columns', column_names)
                # Filter the DataFrame based on the selected data types
                selected_data = car[['price'] + selected_columns]
                st.write('Selected Data: ' + str(selected_data.shape[0]) + ' rows and ' + str(selected_data.shape[1]) + ' columns.')
                if selected_columns:
                    num_charts = len(selected_columns)
                    num_cols = 4  # Number of columns in each row
                    num_rows = (num_charts + num_cols - 1) // num_cols

                    # Create layout for displaying charts in multiple columns
                    chart_layout = st.columns(num_cols)
                    chart_idx = 0
                    
                    for column in selected_columns:
                    # Create a bar chart for each selected column against the "price" column
                        with chart_layout[chart_idx % num_cols]:
                            avg_price_by_feature = car.groupby(column)["price"].mean().reset_index()
                            fig = go.Figure(data=[go.Bar(x=avg_price_by_feature[column], y=avg_price_by_feature["price"])])
                            fig.update_layout(
                                title=f"Average Price vs {column}",
                                xaxis_title=column,
                                yaxis_title="Average Price",
                                width=400,  # Set the width of the plot (adjust as needed)
                                height=300,  # Set the height of the plot (adjust as needed)
                            )
                            st.plotly_chart(fig, use_container_width=True)  # Use container width
                        chart_idx += 1
                else:
                    st.write("Please select at least one column to know the price of selected feature(s).")
                
            
            
            elif selected_option == 'Other Features':
                # Create a dropdown to select the features for the pie chart
                feature_options = [col for col in car.columns if col not in ['price', 'car_ID', 'symboling', 'CompanyName']]
                selected_columns = st.selectbox("Select Feature column(s):", feature_options)
                
                if selected_columns:
                    col1, col2, col3  = st.columns(3)
                    histogram = col1.checkbox("Histogram")
                    pie = col2.checkbox("Pie chart")
                    bar = col3.checkbox("Bar graph")
                                    
                    if histogram:
                        # Histogram
                    
                        # Convert the Matplotlib histogram to a Plotly histogram
                        fig = go.Figure(data=[go.Histogram(x=car[selected_columns], nbinsx=10)])
                        fig.update_layout(title_text=f"Histogram of {selected_columns}", xaxis_title=selected_columns, yaxis_title="Frequency")
                        col1.plotly_chart(fig, use_container_width=True)
                        st.set_option('deprecation.showPyplotGlobalUse', False)


                                        
                    if pie:
                        # Generate the pie chart for the selected feature
                        # Generate the pie chart for the selected feature
                        feature_counts = car[selected_columns].value_counts()
                        fig = go.Figure(data=[go.Pie(labels=feature_counts.index, values=feature_counts)])
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(title=f"Distribution of {selected_columns}")
                        col2.plotly_chart(fig, use_container_width=True)
                                    
                    if bar:
                        plt.figure(figsize=(8, 6))
                        # Convert index values to strings
                        feature_counts = car[selected_columns].value_counts()
                        fig = go.Figure(data=[go.Bar(x=feature_counts.index, y=feature_counts)])
                        fig.update_layout(title=f"Bar Graph of {selected_columns}", xaxis_title=selected_columns, yaxis_title='Frequency')
                        col3.plotly_chart(fig, use_container_width=True)


                

        # Create the expander with a maximum width of 800 pixels
        with st.expander("Visualisation"):
            # Create the radio buttons
            selected_option = st.radio("Select an option", ('Intercorrelation', 'Price Vs. Feature', 'Other Features'), index=1, horizontal=True)
            display_option_data(selected_option)

if __name__ == "__main__":
    main()
