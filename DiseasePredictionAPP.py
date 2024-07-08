import streamlit as st
import numpy as np
import pickle
st.title("Disease Prediction App")
model = pickle.load(open('./models/diseasemodel.pkl','rb'))
data_dict = pickle.load(open('./models/disease_dict.pkl','rb'))

symptoms = st.multiselect("Symptoms: ", list(data_dict['symptom_index'].keys()))

if(st.button("Predict Disease")):
    st.info(f"Your symptoms are {symptoms}")

    input_data = [0] * len(data_dict["symptom_index"]) 
    for symptom in symptoms: 
        index = data_dict["symptom_index"][symptom] 
        input_data[index] = 1
        # reshaping the input data and converting it 
        # into suitable format for model predictions 
    input_data_arr = np.array(input_data).reshape(1,-1)
    disease = model.predict(input_data_arr)
    #st.text(disease)
    namedisease = data_dict["predictions_classes"][disease[0]]
    st.success(f"You may be having {namedisease}")
