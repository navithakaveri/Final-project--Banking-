import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import io
import seaborn as sns
import sweetviz as sv
import matplotlib.pyplot as plt
data=pd.read_csv(r"D:\youtube project\train.csv")
df=pd.DataFrame(data)
def about():
    st.title("FINAL PROJECT--COMPREHENSIVE BANKING ANALYTICS")
    
def about():
    
        #st.title("Comprehensive Banking Analytics")
        st.write("STEPS INVOLVED IN THIS PROJECT")
        st.header("STEP1 : DATA COLLECTION AND PREPROCESSING")
        st.header("STEP2 : EDA ANALYSIS")
        st.markdown("*MANUAL EDA ANALYSIS")
        st.markdown("*AUTOMATE EDA PROCESS")
        st.header("STEP3  :  DATA MODELING")
        st.header("STEP4  :  MODEL EVALUATION")
        st.header("STEP5  :  MODEL DEPLOYEMENT")
        st.header("STEP6  :  INSIGHTS")

def show_shape():
    st.write(df.shape) 
def show_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    return s
def show_values(df):
    missing_values_count = df.isnull().sum()
    st.table(missing_values_count)
def descri(df):
    describe=df.describe()
    st.table(describe)


def datacollection():
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
            df_ = pd.read_csv(uploaded_file)
            
            st.write("Preview of the uploaded file:")
            st.dataframe(df_.head())
            st.subheader("Dataframe Shape")
            show_shape()
            
            #  dataframe information
            st.subheader("Dataframe Information")
            st.text(show_info(df))
            
            # missing values
            st.subheader("Missing Values")
            show_values(df)
            
            st.subheader("Statistics view")
            descri(df)
            
            
def univariate():
    st.title("UNIVARIATE ANALYSIS")
    st.markdown("Analyzing/visualizing the dataset by taking one variable at a time")
    tab1,tab2=st.tabs(["Numerical analysis","Categorical analysis"])
    with tab2:
        st.title("CATEGORICAL VARIABLE ANALYSIS")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.title("CREDIT SCORE")
        sns.barplot(df['Credit_Score'])
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
         categorical_cols=df.select_dtypes(include=['object']).columns
         numumerical_cols = df.select_dtypes(include=np.number).columns.tolist()
         
         for numumerical_cols  in numumerical_cols:
            print('Skew :', round(df[numumerical_cols].skew(), 2))
            plt.figure(figsize = (15, 4))
            plt.subplot(1, 2, 1)
            df[numumerical_cols].hist(grid=False)
            plt.ylabel('count')
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[numumerical_cols])
            st.pyplot()
          
def bivariate():
    tab1,tab2=st.tabs(["Numerical analysis","Categorical analysis"])
    with tab2:
        st.title("")
        plt.figure(figsize=(16, 8))
        sns.scatterplot(data=df, y="Month", x="Num_of_Loan")
        plt.title('MONTH VS NUM_OF_LOAN')
        st.pyplot()
        
        plt.figure(figsize=(13,17))
        sns.pairplot(data=df.drop(['Month','Credit_Score'],axis=1))
        st.pyplot()
        
        
        plt.figure(figsize=(16, 8))
        sns.lineplot(data=df, y="Month", x="Credit_Score")
        plt.title('MONTH vs CREDIT_SCORE')
        st.pyplot()
        
        plt.figure(figsize=(16, 8))
        sns.violinplot(data=df, y="Payment_Behaviour", x="Occupation")
        plt.title('MONTH vs CREDIT_SCORE')
        st.pyplot()
    with tab1:   
        #Plotting a pair plot for bivariate analysis
        g = sns.PairGrid(df,vars=['Interest_Rate','Annual_Income','Num_of_Loan','Total_EMI_per_month','Monthly_Balance'])
        #setting color
        g.map_upper(sns.scatterplot, color='crimson')
        g.map_lower(sns.scatterplot, color='limegreen')
        g.map_diag(plt.hist, color='orange')
        #show figure
        st.pyplot()
        
def multivariate():
      
            
      
        fig = plt.figure(figsize=(25,25))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        non_numeric_cols_to_drop = ['ID', 'Annual_Income', 'Customer_ID', 'Num_Bank_Accounts', 'Num_Credit_Card','Interest_Rate',]
        df_numeric = df.drop(non_numeric_cols_to_drop, axis=1)

        # Convert columns to numeric
        df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

        # Plot correlation heatmap
    
        heatmap = sns.heatmap(df_numeric.corr(), annot=True, cmap='RdPu',annot_kws={"size": 10})
        st.pyplot(heatmap.figure)

def cluster():
    data1=pd.read_csv("C:/Users/navit/Downloads/Bank_cluster1")
    df1=pd.DataFrame(data1)
    
    numeric_columns = df1.select_dtypes(include=['int64','float64']).columns
    categorical_columns = df1.select_dtypes(include=['object']).columns
    
    plt.figure(figsize=(16, 8))
    sns.scatterplot(data=df1, x="cluster", y="Occupation")
    plt.title('MONTH vs CREDIT_SCORE')
    st.pyplot()
    #st.write("NUMERICAL COLUMNS",numeric_columns)
    #st.write("CATEGORICAL_COLUMNS",categorical_columns)
    Occupation=df1['Occupation'].unique()
    Credit_Score=df1['cluster'].unique()
    with st.form("my_form"):
        Occupation = st.selectbox("Occupation", Occupation,key=1)
        
        Credit_Score = st.selectbox("Credit_Score", Credit_Score,key=2)
        submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
        
        
        
    
   
        
        

     
 
      
                
    
        
            
            
            
        
        
            
     

           
            # Data preprocessing
def main():
    
    st.set_page_config(layout="wide",page_title="COMPREHENSIVE BANKING ANALYTICS")
    page=option_menu(" ",["ABOUT","DATA COLLECTION & PREPROCESSING","MANUAL EDA PROCESS","AUTO EDA PROCESS","INSIGHTS"],orientation='horizontal')
    if page=="ABOUT":
        about()
    elif page=="DATA COLLECTION & PREPROCESSING":
        datacollection()
    elif page=="MANUAL EDA PROCESS":
       
            page =st.radio("SELECT",["UNIVARIATE_ANALYSIS", "BIVARIATE_ANALYSIS","MULTIVARAIATE_ANALYSIS"])
            if page=="UNIVARIATE_ANALYSIS":
                univariate()
            elif page=="BIVARIATE_ANALYSIS":
                bivariate()
            elif page=="MULTIVARAIATE_ANALYSIS":
                multivariate()
    elif page=="AUTO EDA PROCESS":
            st.title("Auto EDA process")
      # Perform automated EDA using Sweetviz
            report = sv.analyze(df)
      # Save the Sweetviz report as an HTML file
            report_file_path = "sweetviz_report.html"
            report.show_html(report_file_path)
        # Read the HTML file
            with open(report_file_path, "r", encoding="utf-8") as f:
                 html_content = f.read()

        # Display the HTML content within the Streamlit app
            st.components.v1.html(html_content, height=1000, scrolling=True)
            
        
    elif page=="INSIGHTS":
        cluster()
    
if __name__ == "__main__":
        main()
          