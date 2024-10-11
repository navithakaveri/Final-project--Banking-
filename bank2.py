import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import io
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
import pickle
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide", page_title="COMPREHENSIVE BANKING ANALYTICS")

# Load your dataset here
data = pd.read_csv(r"D:\youtube project\Bank_class")
df = pd.DataFrame(data)

# Define the About Section
def about():
    st.title("FINAL PROJECT--COMPREHENSIVE BANKING ANALYTICS")
    st.subheader("Project Overview")
    st.write("""
    This project focuses on predicting loan availability based on customer financial data using a Decision Tree classifier. 
    It demonstrates skills in data preprocessing, model training, and model evaluation, including accuracy scores, 
    confusion matrix, and classification reports.
    """)

# Function to display the shape of the dataset
def show_shape():
    st.write(df.shape)

# Function to display the info of the dataset
def show_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    return s

# Function to display missing values
def show_values(df):
    missing_values_count = df.isnull().sum()
    st.table(missing_values_count)

# Function to display descriptive statistics
def descri(df):
    describe = df.describe()
    st.table(describe)

# Data Collection Section
def datacollection():
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df_ = pd.read_csv(uploaded_file)
        
        st.write("Preview of the uploaded file:")
        st.dataframe(df_.head())
        st.subheader("Dataframe Shape")
        show_shape()
        
        st.subheader("Dataframe Information")
        st.text(show_info(df))
        
        st.subheader("Missing Values")
        show_values(df)
        
        st.subheader("Statistics view")
        descri(df)

# Univariate Analysis Section
st.set_option('deprecation.showPyplotGlobalUse', False)           
def univariate():
    st.title("UNIVARIATE ANALYSIS")
    st.markdown("Analyzing/visualizing the dataset by taking one variable at a time")
    tab1, tab2 = st.tabs(["Numerical analysis", "Categorical analysis"])
    
    with tab2:
        order_data = df['Type_of_Loan'].value_counts().iloc[:20].index
        plt.figure(figsize=(15, 10))
        sns.countplot(y='Type_of_Loan', data=df, order=order_data)
        st.title('Type_of_Loan')
        plt.xlabel('count')
        plt.ylabel('Type_of_Loan')
        st.pyplot()
        
        credit_score_counts = df['Credit_Score'].value_counts().iloc[:20]
        plt.figure(figsize=(10, 7))
        plt.pie(credit_score_counts, labels=credit_score_counts.index, autopct='%1.1f%%', startangle=90)
        st.title('Credit Score Distribution')
        plt.axis('equal')
        st.pyplot()
        
        st.title("Payment_Behaviour")
        sns.countplot(df['Payment_Behaviour'])
        st.pyplot()
        
        st.title("Credit_Mix")
        sns.countplot(df['Credit_Mix'])
        st.pyplot()
        
        st.title("Occupation")
        sns.countplot(df['Occupation'])
        st.pyplot()
        
    with tab1:
        st.title("NUMERICAL VARIABLE ANALYSIS")
        numumerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        for numumerical_col in numumerical_cols:
            st.write(f'Analyzing {numumerical_col}')
            plt.figure(figsize=(15, 4))
            plt.subplot(1, 2, 1)
            df[numumerical_col].hist(grid=False)
            plt.ylabel('count')
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[numumerical_col])
            st.pyplot()

# Bivariate Analysis Section
def bivariate():
    tab1, tab2 = st.tabs(["Numerical analysis", "Categorical analysis"])
    with tab2:
        sns.barplot(df, y='Payment_Behaviour', x='Num_of_Delayed_Payment')
        plt.title('Payment_Behaviour vs Num_of_Delayed_Payment')
        st.pyplot()
        
        sns.barplot(df, y='Credit_Score', x='Annual_Income')
        st.pyplot()
        
        sns.barplot(df, y='Credit_Score', x='Num_of_Delayed_Payment')
        plt.title('Credit Score vs Num_of_Delayed_Payment')
        st.pyplot()
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df['Annual_Income'], y=df['Monthly_Inhand_Salary'])
        plt.title('Annual Income vs Monthly Inhand Salary')
        st.pyplot()
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=df['Occupation'], y=df['Annual_Income'])
        plt.title('Occupation vs Annual Income')
        plt.xticks(rotation=45)
        st.pyplot()
        
        pd.crosstab(df['Credit_Score'], df['Occupation']).plot(kind='bar', figsize=(12, 6))
        plt.title('Occupation vs Credit Score')
        st.pyplot()
        
        pd.crosstab(df['Credit_Score'], df['Payment_Behaviour']).plot(kind='bar', figsize=(12, 6))
        plt.title('Payment Behaviour vs Credit Score')
        st.pyplot()

    with tab1:
        g = sns.PairGrid(df, vars=['Interest_Rate', 'Annual_Income', 'Num_of_Loan', 'Total_EMI_per_month', 'Monthly_Balance'])
        g.map_upper(sns.scatterplot, color='crimson')
        g.map_lower(sns.scatterplot, color='limegreen')
        g.map_diag(plt.hist, color='orange')
        st.pyplot()

# Multivariate Analysis Section
def multivariate():
    categorical_data = df.select_dtypes(include=[object]).columns
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    plt.figure(figsize=(15, 10))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', linewidths=0.1)
    plt.title('Correlation Matrix')
    st.pyplot()

    fig = plt.figure(figsize=(25, 25))
    non_numeric_cols_to_drop = ['ID', 'Annual_Income', 'Customer_ID', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate']
    df_numeric = df.drop(non_numeric_cols_to_drop, axis=1)
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')
    heatmap = sns.heatmap(df_numeric.corr(), annot=True, cmap='RdPu', annot_kws={"size": 10})
    st.pyplot(heatmap.figure)

# Clustering Section
def cluster():
    data1 = pd.read_csv("C:/Users/navit/Downloads/Bank_cluster1")
    df1 = pd.DataFrame(data1)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Credit_Score', y='Occupation', hue='cluster', data=df1, palette='viridis')
    plt.title('Credit Score by Occupation')
    st.pyplot()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(y='Payment_Behaviour', x='Occupation', hue='cluster', data=df1, palette='viridis')
    plt.title('Payment Behaviour Cluster')
    st.pyplot()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(y='Outstanding_Debt', x='Occupation', hue='cluster', data=df1, palette='viridis')
    plt.title('Outstanding Debt Cluster')
    st.pyplot()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Credit_Score', y='Annual_Income', hue='cluster', data=df1, palette='viridis')
    plt.title('Annual Income Cluster')
    st.pyplot()



# Title of the app
st.title("Loan Availability Prediction")

def labeling(late_payment):
        if late_payment<=2:
            return 'low_risk'
        elif 3 <= late_payment <=10 :
            return 'medium_risk'
        else:
            return 'high_risk'
        
def loan(x,y):
        if ( x in ['Standard','Good'] ) and (y in ['low_risk','medium_risk']):
            return 'Approved'
        else:
            return 'Rejected'  

# Function to handle the prediction
def classification_model():
    st.subheader("Classification Model")
    
    # Sample dropdowns for categorical inputs
    Occupation = ['Management', 'Labourer', 'Clerical', 'Business']
    Credit_Mix = ['Standard', 'Good', 'Bad']
    Payment_of_Min_Amount = ['No', 'Yes']
    Payment_Behaviour = ['High_spent_Small_value_payments', 'Low_spent_Large_value_payments',
                         'Low_spent_Small_value_payments', 'High_spent_Large_value_payments']

    # Form for user inputs
    with st.form("my_form"):
        Age = st.slider("Age", min_value=18, max_value=80, value=30, step=1)
        Occupation = st.selectbox("Occupation", options=Occupation)
        Num_Bank_Accounts = st.slider("Number of Bank Accounts", min_value=0, max_value=20, value=5, step=1)
        Num_Credit_Card = st.slider("Number of Credit Cards", min_value=0, max_value=20, value=5, step=1)
        Interest_Rate = st.slider("Interest Rate", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
        Num_of_Loan = st.slider("Number of Loans", min_value=0, max_value=20, value=2, step=1)
        Num_of_Delayed_Payment = st.slider("Number of Delayed Payments", min_value=0, max_value=100, value=5, step=1)
        Num_Credit_Inquiries = st.slider("Number of Credit Inquiries", min_value=0, max_value=10, value=2, step=1)
        Credit_Mix = st.selectbox("Credit Mix", options=Credit_Mix)
        Payment_of_Min_Amount = st.selectbox("Payment of Minimum Amount", options=Payment_of_Min_Amount)
        Payment_Behaviour = st.selectbox("Payment Behaviour", options=Payment_Behaviour)
        Annual_Income = st.slider("Annual Income", min_value=10000, max_value=1500000, value=50000, step=10000)
        Monthly_Inhand_Salary= st.slider("Monthly Inhand Salary", min_value=0, max_value=500000, value=20000, step=1000)
        Delay_from_due_date = st.slider("Delay from Due Date (Days)", min_value=0, max_value=100, value=5, step=1)
        Changed_Credit_Limit = st.slider("Changed Credit Limit", min_value=0, max_value=50000, value=10000, step=1000)
        Outstanding_Debt = st.slider("Outstanding Debt", min_value=0, max_value=1000000, value=50000, step=10000)
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            # Prepare the input for the model
            input_data = {
                'Age': Age,
                'Occupation': Occupation,
                'Num_bank_acc': Num_Bank_Accounts,
                'Num_credit_card': Num_Credit_Card,
                'Interest_rate': Interest_Rate,
                'Num_of_loans': Num_of_Loan,
                'Num_of_Delayed_Payment': Num_of_Delayed_Payment,
                'Num_Credit_Inquiries': Num_Credit_Inquiries,
                'Credit_mix': Credit_Mix,
                'Payment_of_Min_Amount': Payment_of_Min_Amount,
                'Payment_Behaviour': Payment_Behaviour,
                'Annual_Income': Annual_Income,
                'Monthly_Inhand_Salary': Monthly_Inhand_Salary,
                'Delay_from_due_date': Delay_from_due_date,
                'Changed_Credit_Limit': Changed_Credit_Limit,
                'Outstanding_Debt': Outstanding_Debt
            }
            
            # Convert input into DataFrame
            df_input = pd.DataFrame([input_data])
            
            # Encoding categorical data
            label_encoders = {}
            categorical_columns = ['Occupation', 'Credit_mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
            for col in categorical_columns:
                le = LabelEncoder()
                df_input[col] = le.fit_transform(df_input[col])
                label_encoders[col] = le  # Store encoders if you need to reverse the transformation later
                print("Input shape:", df_input.shape)

            #model = load('final_class_pre.joblib', mmap_mode='r')
            model = load('final_class_pre.joblib')
            predicted = model.predict(df_input)  
            #Risk Assesment
            Risk_Assessment = labeling(Num_of_Delayed_Payment)

            # mapping the loan status based on the predicted value
            Loan_Status = loan(predicted,Risk_Assessment)
            
            if submitted:
                    predicted_data = "Loan Availability Status: " + Loan_Status
                    st.success(predicted_data)
  


def main():
            
# Sidebar Navigation
    with st.sidebar:
        selected = option_menu("Main Menu", ["ABOUT", "DATA COLLECTION", "UNIVARIATE","BIVARIATE","MULTIVARIATE","CLUSTER","CLASSIFICATION MODEL"],
                            icons=['house', 'cloud-upload', 'bar-chart', 'graph-up', 'layers', 'check-circle'], 
                            menu_icon="cast", default_index=0)

    # Display sections based on user selection
    if selected == "ABOUT":
        about()
    elif selected == "DATA COLLECTION":
        datacollection()
    elif selected == "UNIVARIATE":
        univariate()
    elif selected == "BIVARIATE":
        bivariate()
    elif selected == "MULTIVARIATE":
        multivariate()
    elif selected == "CLUSTER":
        cluster()
    elif selected == "CLASSIFICATION MODEL":
        classification_model()
            
if __name__ == "__main__":
    main()
        
