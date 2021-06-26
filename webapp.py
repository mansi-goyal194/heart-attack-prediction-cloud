import streamlit as st
import pickle

log_model = pickle.load(open('log_model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
def classify(num):
    if num<0.9:
        return 'No risk of heart attack'
    elif num>0.9:
        print(num)
        return 'Risky'
        
    else:
        return 'HEHE'

def main():
    st.title("Heart Attack Prediction")
    html_temp = """
    <div style = "background-color:teal; padding:10px;">
    <h2 style = "color:white ; text-align:center; "> Heart Attack Prediction</h2>
    </div>
    """ 
    st.markdown(html_temp, unsafe_allow_html=True)
    activities = ['Logistic Regression']
    options = st.sidebar.selectbox('Which model would you like to choose', activities)
    st.subheader(options)
    st.spinner('Hello')
    age=st.slider('Select Age',1,100)
    sex=st.selectbox('Select Gender',(0,1))
    cp=st.selectbox('Select Chest pain level',(1,2,3,4))
    restbp=st.slider('Select resting blood pressure',0,200)
    chol=st.slider('Select cholestrol level',100,300)
    fbs=st.selectbox('Select fasting blood sugar level,(1 for fbs>120)',(0,1))
    restecg=st.selectbox('Select Resting electrocardiographic results',(0,1,2))
    thalach=st.slider('Select maximum Heart rate achieved',0,200)
    exang=st.selectbox('Select exercise induced angina',(0,1))
    oldpeak=st.slider('Select ST depression induced',0.0,10.0)
    slope=st.selectbox('Select slope of peak exercise',(1,2,3))
    ca=st.selectbox('Select number of major vessels',(0.0,1.0,2.0,3.0))
    thal=st.selectbox('Select thal defect',(3.0,6.0,7.0))

    inputs=[[age,sex,cp,restbp,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]
    for i in range(12):
        inputs[0][i] = inputs[0][i] - scaler.mean_[i]
        print(inputs)
    if st.button('Classify'):
        if options=='Logistic Regression':
            prediction = log_model.predict(inputs)
            print(prediction)
            st.success(classify(prediction))

if __name__ == '__main__':
    main()

