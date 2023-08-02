import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
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
    page_icon="car.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.title("Car Price Prediction Model")
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

        st.subheader("Regression Model Evaluation Results")
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

                if Heatmap:
                    st.write('##### Intercorrelation Matrix Heatmap :')
                    numeric_columns = car.select_dtypes(include=[np.number])
                    numeric_columns = numeric_columns.drop(columns=['car_ID', 'symboling'])
                    corr = numeric_columns.corr()
                    with sns.axes_style("white"):
                        fig, ax = plt.subplots(figsize=(7, 5))
                        ax = sns.heatmap(corr, vmax=1, square=True)
                    st.pyplot(fig)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                # Display the correlation matrix
                if Matrix:
                    numerical_cols = car.select_dtypes(include=[np.number]).columns
                    correlation_matrix = car[numerical_cols].corr()
                    st.write("##### Correlation Matrix :")
                    st.dataframe(correlation_matrix)
              
            if selected_option == 'Price Vs. Feature':
                exclude_columns = ['symboling', 'car_ID', 'price']  # Replace with the actual column names you want to exclude
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
                            
                            plt.figure(figsize=(8, 5))
                            ax = sns.barplot(x=car[column], y=car["price"])
                            ax.set_xlabel(column)
                            ax.set_ylabel("Price")
                            ax.set_title(f"{column} vs Price")
                            st.pyplot(plt.gcf())

                        chart_idx += 1
                else:
                    st.write("Please select at least one column to know the price of selected feature(s).")
                
            elif selected_option == 'Other features':
                col1, col2, col3,col4 = st.columns(4)
                checkbox1 = col1.checkbox("Frequency of Cars sold")
                checkbox2 = col1.checkbox("Price distribution of cars")
                checkbox3 = col2.checkbox("Fuel type Ratio")
                checkbox4 = col2.checkbox("Gas & Diesel vehicles")
                checkbox5 = col3.checkbox("Aspiration")
                checkbox6 = col3.checkbox("Turbo and Std aspiration")
                checkbox7 = col4.checkbox("Door Number")
                checkbox8 = col4.checkbox("Carbody")
                checkbox9 = col1.checkbox("Engine type")
                checkbox10 = col2.checkbox("Cylinder number")
                checkbox11 = col3.checkbox("Fuel system")
                checkbox12 = col4.checkbox("Car Length vs. Car Width")
                if checkbox1:
                    st.write("#### Visualise different car names")
                    plt.figure(figsize=(25, 6))
                    plt.subplot(1,3,1)
                    plt1 = car.CompanyName.value_counts().plot(kind = 'bar', color='olive')
                    plt.title('Frequency of Cars sold in American Markets')
                    plt1.set(xlabel = 'Car Brands', ylabel='Frequency of cars sold')
                    st.pyplot(plt.gcf())
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                if checkbox3:
                    st.write('##### Fuel type Ratio')
                    df=pd.DataFrame(car['fueltype'].value_counts())
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.pie(df['fueltype'], labels=df.index, autopct='%1.1f%%', colors=['steelblue', 'purple'])
                    ax.set_title('Fuel Type Ratio')
                    st.pyplot(fig)
                if checkbox2:
                    st.write('#### Distribution of Car Prices')
                    plt.figure(figsize=(8, 6))
                    sns.distplot(car['price'], kde=True, color='purple')
                    # Adding labels and title
                    plt.xlabel('Price')
                    plt.ylabel('Density')
                    plt.title('Distribution of Car Prices')
                    st.pyplot(plt.gcf())
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                if checkbox4:
                    st.write('#### Distribution of price of gas vehicles and diesel vehicles')
                    f = plt.figure(figsize=(12,5))
                    ax = f.add_subplot(121)
                    sns.distplot(car[car.fueltype == 'gas']['price'], color='gray', ax=ax)
                    ax.set_title('Distribution of price of gas vehicles')
                    ax = f.add_subplot(122)
                    sns.distplot(car[car.fueltype == 'diesel']['price'], color='darkgreen', ax=ax)
                    ax.set_title('Distribution of ages of diesel vehicles')
                    st.pyplot(f)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.write('#### Box plot of Price by Fuel Type')
                    plt.figure(figsize=(8, 6))
                    sns.boxplot(x='fueltype', y='price', data=car, palette='magma')
                    plt.title('Box plot of Price by Fuel Type')
                    st.pyplot(plt.gcf())
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                if checkbox5:
                    # Display the pie chart for the "aspiration" column
                    df_aspiration = pd.DataFrame(car['aspiration'].value_counts())
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.pie(df_aspiration['aspiration'], labels=df_aspiration.index, autopct='%1.1f%%', colors=['cyan', 'orange'])
                    ax.set_title('Aspiration Type Ratio')
                    st.pyplot(fig)
                    st.set_option('deprecation.showPyplotGlobalUse', False)

                if checkbox6:
                    st.write('#### Price distribution of Turbo and Std aspiration vehicles')
                    f = plt.figure(figsize=(12, 5))
                    ax = f.add_subplot(121)
                    sns.distplot(car[car.aspiration == 'turbo']['price'], color='navy', ax=ax)
                    ax.set_title('Price distribution of Turbo aspiration vehicles')
                    ax = f.add_subplot(122)
                    sns.distplot(car[car.aspiration == 'std']['price'], color='orange', ax=ax)
                    ax.set_title('Price distribution of Std aspiration vehicles')
                    st.pyplot(f)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x='aspiration', y='price', data=car, palette='cividis')
                    plt.title('Box plot of Price by Aspiration Type')
                    st.pyplot(plt.gcf())
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    
                if checkbox7:
                    
                     # Calculate the door number distribution
                    df_doornumber = pd.DataFrame(car['doornumber'].value_counts())

                    # Create a pie chart for door number distribution
                    plt.figure(figsize=(5, 5))
                    plt.pie(df_doornumber['doornumber'], labels=df_doornumber.index, autopct='%1.1f%%', colors=['m', 'y'])
                    plt.title('Door Number Distribution')
                    st.pyplot(plt.gcf())
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    f = plt.figure(figsize=(12, 5))
                    # Price distribution of cars having two doors
                    ax = f.add_subplot(121)
                    sns.distplot(car[car.doornumber == 'two']["price"], color='darkgoldenrod', ax=ax)
                    ax.set_title('Price distribution of cars having two doors')

                    ax = f.add_subplot(122)
                    sns.distplot(car[car.doornumber == 'four']['price'], color='darkblue', ax=ax)
                    ax.set_title('Price distribution of cars having four doors')

                    st.pyplot(f)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                
                if checkbox8:
                    # Calculate the Carbody distribution
                    df_carbody = pd.DataFrame(car['carbody'].value_counts())

                    # Create a pie chart for Carbody distribution
                    plt.figure(figsize=(8, 8))
                    plt.pie(df_carbody['carbody'], labels=df_carbody.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
                    plt.title('Carbody Distribution')
                    st.pyplot(plt.gcf())
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                # Price distribution according to car body
                    plt.figure(figsize=(8, 6))
                    sns.boxplot(x='carbody', y='price', data=car, palette='hot')
                    plt.title('Price Distribution According to Car Body')
                    st.pyplot(plt.gcf())
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                
                if checkbox9:
                    # Calculate the Engine type distribution
                    df_enginetype = pd.DataFrame(car['enginetype'].value_counts())

                    # Create a pie chart for Engine type distribution
                    plt.figure(figsize=(8, 8))
                    plt.pie(df_enginetype['enginetype'], labels=df_enginetype.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
                    plt.title('Engine Type Distribution')
                    st.pyplot(plt.gcf())
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    # Price distribution according to engine type
                    plt.figure(figsize=(8, 6))
                    sns.boxplot(x='enginetype', y='price', data=car, palette='Accent')
                    plt.title('Price Distribution According to Engine Type')
                    st.pyplot(plt.gcf())
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                
                if checkbox10:
                    # Calculate the Cylinder number distribution
                    df_cylindernumber = pd.DataFrame(car['cylindernumber'].value_counts())

                    # Create a pie chart for Cylinder number distribution
                    plt.figure(figsize=(8, 8))
                    plt.pie(df_cylindernumber['cylindernumber'], labels=df_cylindernumber.index, autopct='%1.1f%%', colors=sns.color_palette('cool'))
                    plt.title('Cylinder Number Distribution')
                    st.pyplot(plt.gcf())
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                       # Price distribution according to cylinder number
                    plt.figure(figsize=(8, 6))
                    sns.boxplot(x='cylindernumber', y='price', data=car, palette='autumn')
                    plt.title('Price Distribution According to Cylinder Number')
                    st.pyplot(plt.gcf())
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    
                if checkbox11:
                    # Calculate the Fuel system distribution
                    df_fuelsystem = pd.DataFrame(car['fuelsystem'].value_counts()).reset_index().rename(columns={'index': 'fuelsystem', 'fuelsystem': 'count'})

                    # Create a bar plot for Fuel system distribution
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x='fuelsystem', y='count', data=df_fuelsystem)
                    plt.title('Fuel System Distribution')
                    plt.xlabel('Fuel System')
                    plt.ylabel('Count')
                    st.pyplot(plt.gcf())
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                # Price distribution according to fuel system
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x='fuelsystem', y='price', data=car, palette='gist_rainbow')
                    plt.title('Price Distribution According to Fuel System')
                    plt.xlabel('Fuel System')
                    plt.ylabel('Price')
                    st.pyplot(plt.gcf())
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                
                if checkbox12:
                     # Scatter plot of Car Length vs. Car Width
                        plt.figure(figsize=(10, 6))
                        sns.scatterplot(x="carlength", y="carwidth", data=car, color='b')
                        plt.title("Car Length vs. Car Width")
                        plt.xlabel("Car Length")
                        plt.ylabel("Car Width")
                        st.pyplot(plt.gcf())
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                    # Joint plot of Car Length vs. Car Width
                        plt.figure(figsize=(10, 6))
                        g = sns.jointplot(x="carwidth", y="carlength", data=car, kind="kde", color="pink")
                        g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
                        g.ax_joint.collections[0].set_alpha(0)
                        g.set_axis_labels("Car Width", "Car Length")
                        st.pyplot(plt.gcf())
                        st.set_option('deprecation.showPyplotGlobalUse', False)
            
            
            elif selected_option == 'Generic Features':
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
                        plt.figure(figsize=(8, 6))
                        plt.hist(car[selected_columns], bins=10, edgecolor='black')
                        plt.xlabel(selected_columns)
                        plt.ylabel('Frequency')
                        plt.title(f"Histogram of {selected_columns}")
                        st.pyplot(use_container_width=True)
                        
                   # Generate the pie chart for the selected feature
                    if pie:
                        feature_counts = car[selected_columns].value_counts()
                        plt.figure(figsize=(8, 6))
                        plt.pie(feature_counts, labels=feature_counts.index, autopct="%1.1f%%", startangle=140)
                        plt.axis('equal')
                        plt.title(f"Distribution of {selected_columns}")
                        st.pyplot(use_container_width=True)
                    if bar:
                        plt.figure(figsize=(8, 6))
                         # Convert index values to strings
                        feature_counts = car[selected_columns].value_counts()
                        feature_labels = [str(label) for label in feature_counts.index]
                        plt.bar(feature_labels, feature_counts)
                        plt.xlabel(selected_columns)
                        plt.ylabel('Frequency')
                        plt.title(f"Bar Graph of {selected_columns}")
                        plt.xticks(rotation=45)
                        st.pyplot(use_container_width=True)
                   
                

        # Create the expander with a maximum width of 800 pixels
        with st.expander("Visualisation"):
            # Create the radio buttons
            selected_option = st.radio("Select an option", ('Intercorrelation', 'Price Vs. Feature', 'Generic Features'), index=1, horizontal=True)
            display_option_data(selected_option)

if __name__ == '__main__':
    main()
