import streamlit as st
import sklearn
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

pipe=pickle.load(open('pipe.pkl','rb'))

st.title('Customer Churn Predictor')

col1,col2=st.columns(2)

with col1:
    Voice_Plan=st.number_input('Select Voice plan is active or not')

with col2:
    Voice_Messages=st.number_input('Number of voicemail messages')

col3,col4=st.columns(2)

with col3:
    International_plan=st.number_input('International plan is active or not')

with col4:
    International_min=st.number_input('How many minutes customer used service to make international calls')    

col5,col6=st.columns(2)

with col5:
    International_calls=st.number_input('Total number of international calls') 

with col6:
    International_charge=st.number_input('Total international charge') 

col7,col8=st.columns(2)

with col7:
    Customer_calls=st.number_input('Total number of calls to customer service') 
    
with col8:
    Total_Days_Mins=st.number_input('Total number of minutes customer used service during the day')

col9,col10=st.columns(2)

with col9:
    Total_Charge=st.number_input('Total charge for calls') 

with col10:
    Total_Days_Calls=st.number_input('Total number of Call in a Day')

if st.button('Predict Probability'):

        data={
        'voice.plan':[Voice_Plan],
        'voice.messages':[Voice_Messages],
        'intl.plan':[International_plan],
        'intl.mins':[International_min],
        'intl.calls':[International_calls],
        'intl.charge':[International_charge],
        'customer.calls':[Customer_calls],
        'TotalDaysMins':[Total_Days_Mins],
        'TotalCharge':[Total_Charge],
        'TotalDaysCalls':[Total_Days_Calls],
           
        }

        input_df=pd.DataFrame(data)

        result=pipe.predict_proba(input_df)

        losspro=result[0][0]
        winpro=result[0][1]

        st.header('Chances of customer Staying:-'+str(round(winpro*100))+'%')
        st.header('Chances of customer leaving:-'+str(round(losspro*100))+'%')
        

                    
            
           
           
    
       
