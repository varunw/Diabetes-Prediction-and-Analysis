import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pandas_profiling as pp
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
from  streamlit_pandas_profiling import st_profile_report

model=pickle.load(open('Logistic_Regression_Pickle_one.pickle','rb'))
dataset=pd.read_csv("pima-data.csv")

count = 0

for i in range(768):
    if dataset["glucose_conc"][i] == np.int64(0):
        count = count + 1
    if dataset["diastolic_bp"][i] == np.int64(0):
        count = count + 1
    if dataset["thickness"][i] == np.int64(0):
        count = count + 1
    if dataset["insulin"][i] == np.int64(0):
        count = count + 1
    if dataset["bmi"][i] == np.int64(0):
        count = count + 1
    if dataset["diab_pred"][i] == np.int64(0):
        count = count + 1
    if dataset["age"][i] == np.int64(0):
        count = count + 1
    if dataset["skin"][i] == np.int64(0):
        count = count + 1

size = dataset.size
shape = dataset.shape
percent = (count / size) * 100

def predict_diabetes(pregnancies,glucose,bp,thickness,bmi,diab_pred,age,skin):
    input=np.array([[pregnancies,glucose,bp,thickness,bmi,diab_pred,age,skin]]).astype(np.float64)
    prediction=model.predict(input)
    print(prediction)
    #pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return prediction

def main():
    st.title("Diabetes_Prediction")
    data = dataset.drop("insulin", axis=1)
    data = data.dropna()
    data = data[-(data[data.columns[1:-1]] == 0).any(axis=1)]


    df = pd.DataFrame({"Dataset": ["Number of variables", "Number of Observations", "Number of Missing cells", "Number of missing cells %"],"Results":[shape[1],size,count,percent]})
    if st.sidebar.checkbox("Data Description"):
        st.write(dataset)
        profile=ProfileReport(dataset)
        st_profile_report(profile)


    if st.sidebar.checkbox("Predict"):
        st.text("Helli")
        html_temp = """
            <div style="background-color:#025246 ;padding:10px">
            <h2 style="color:white;text-align:center;">Diabetes Prediction ML App </h2>
            </div>
            """
        st.markdown(html_temp, unsafe_allow_html=True)

        pregnancies = st.text_input("pregnancies", "Type Here")
        glucose = st.text_input("glucose", "Type Here")
        bp = st.text_input("Blood Pressure", "Type Here")
        thickness = st.text_input("thickness", "Type Here")
        #insulin = st.text_input("Insulin", "Type Here")
        bmi = st.text_input("BMI", "Type Here")
        diab_pred = st.text_input("Diabetes_Pred", "Type Here")
        age = st.text_input("Age", "Type Here")
        skin = st.text_input("Skin", "Type Here")
        safe_html = """  
              <div style="background-color:#F4D03F;padding:10px >
               <h2 style="color:white;text-align:center;"> Your forest is safe</h2>
               </div>
            """
        danger_html = """  
              <div style="background-color:#F08080;padding:10px >
               <h2 style="color:black ;text-align:center;"> Your forest is in danger</h2>
               </div>
            """

        if st.button("Predict"):
            output = predict_diabetes(pregnancies, glucose, bp, thickness, bmi, diab_pred, age, skin)

            if output[0] == np.bool_(True):
                st.success("The patient has Diabetes")
            else:
                st.success("The patient does not have Diabetes")

            # print(output)

            # if output > 0.5:
            #    st.markdown(danger_html,unsafe_allow_html=True)
            # else:
            #    st.markdown(safe_html,unsafe_allow_html=True)

        result = model.predict([[0, 100, 88, 30,  32.5, 0.855, 38, 1.183]])
        print(type(result[0]))
        if (result == np.bool_(True)):
            print("Numpy OP")

        # input = np.array([[0, 100, 88, 30, 0, 32.5, 0.855, 38, 1.183]]).astype(np.float64)
        # prediction = model.predict_proba(input)
        # print(prediction)

    if st.sidebar.checkbox("Pairplot"):
       fig = sns.pairplot(data, hue='diabetes', diag_kind='hist')
       st.pyplot(fig)

    if st.sidebar.checkbox("2D Histogram"):
        plt.figure(figsize=(10, 9))  # 2D Histogram
        plt.hist2d(data['glucose_conc'], data['bmi'], bins=(20, 20), cmap='magma')
        plt.xlabel("glucose")
        plt.ylabel('bmi')
        plt.colorbar()
        st.pyplot()

    if st.sidebar.checkbox("DistPlot"):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        for i, col in enumerate(data.columns[:-1]):
            plt.figure(i)
            sns.distplot(data[col],rug=True)
            st.pyplot()

    if st.sidebar.checkbox('Pie Chart'):
        st.set_option('deprecation.showPyplotGlobalUse',False)
        fig1, ax1 = plt.subplots(1, 2, figsize=(8, 8))
        sns.countplot(data['diabetes'], ax=ax1[0])
        labels = 'Diabetic', 'Healthy'
        data.diabetes.value_counts().plot.pie(labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        st.pyplot()

    if st.sidebar.checkbox('Plots'):
        sns.pointplot(dataset['num_preg'], dataset['age'], hue=dataset['diabetes'])
        st.pyplot()
        sns.pointplot(dataset['glucose_conc'], dataset['bmi'], hue=dataset['diabetes'])
        st.pyplot()
        sns.jointplot(dataset['num_preg'], dataset['age'], kind='hex')
        st.pyplot()
        sns.lmplot(x='num_preg', y='age', data=data, hue='diabetes')
        st.pyplot()

    if st.sidebar.checkbox('BoxPlots'):
        sns.boxplot(x="diabetes", y="num_preg", data=data, whis=3.0);
        st.pyplot()
        sns.boxplot(x="diabetes", y="glucose_conc", data=data, whis=3.0);
        st.pyplot()
        sns.boxplot(x="diabetes", y="diastolic_bp", data=data, whis=3.0);
        st.pyplot()
        sns.boxplot(x="diabetes", y="thickness", data=data, whis=3.0);
        st.pyplot()
        sns.boxplot(x="diabetes", y="bmi", data=data, whis=3.0);
        st.pyplot()
        sns.boxplot(x="diabetes", y="diab_pred", data=data, whis=3.0);
        st.pyplot()
        sns.boxplot(x="diabetes", y="age", data=data, whis=3.0);
        st.pyplot()
        sns.boxplot(x="diabetes", y="skin", data=data, whis=3.0);
        st.pyplot()
        sns.boxplot(x="diabetes", y="bmi", data=data, whis=3.0);
        st.pyplot()


if __name__=='__main__':
    main()