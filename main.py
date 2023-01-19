def main():
    # take input from users using st.form function
       with st.form("road_traffic_severity_form"):
              st.subheader("Pleas enter the following inputs:")
              # define variable to store user inputs
              No_vehicles = st.slider("Number of vehicles involved:",1,7, value=0, format="%d")
              No_casualties = st.slider("Number of casualities:",1,8, value=0, format="%d")
              Hour = st.slider("Hour of the day:", 0, 23, value=0, format="%d")
              collision = st.selectbox("Type of collision:",options=options_types_collision)
              Age_band = st.selectbox("Driver age group?:", options=options_age)
              Sex = st.selectbox("Sex of the driver:", options=options_sex)
              Education = st.selectbox("Education of driver:",options=options_education_level)
              service_vehicle = st.selectbox("Service year of vehicle:", options=options_services_year)
              Day_week = st.selectbox("Day of the week:", options=options_day)
              Accident_area = st.selectbox("Area of accident:", options=options_acc_area)
              
              # put a submit button to predict the output of the model
              submit = st.form_submit_button("Predict")

       if submit:
              input_array = np.array([collision,
                                   Age_band,Sex,Education,service_vehicle,
                                   Day_week,Accident_area], ndmin=2)

              # encode all the categorical features 
              encoded_arr = list(encoder.transform(input_array).ravel())
              
              num_arr = [No_vehicles,No_casualties,Hour]
              pred_arr = np.array(num_arr + encoded_arr).reshape(1,-1)              
              # predict the output using model object
              prediction = model.predict(pred_arr)
              
              if prediction == 0:
                     st.write(f"The severity prediction is Fatal Injury⚠")
              elif prediction == 1:
                     st.write(f"The severity prediction is serious injury")
              else:
                     st.write(f"The severity prediciton is slight injury")

              # Explainable AI using shap library for model explanation.
              st.subheader("Explainable AI (XAI) to understand predictions")  
              shap.initjs()
              shap_values = shap.TreeExplainer(model).shap_values(pred_arr)
              st.write(f"For prediction {prediction}") 
              shap.force_plot(shap.TreeExplainer(model).expected_value[0], shap_values[0],
                              pred_arr, feature_names=features, matplotlib=True,show=False).savefig("pred_force_plot.jpg", bbox_inches='tight')
              img = Image.open("pred_force_plot.jpg")
              # image to display on website 
              st.image(img, caption='Model explanation using shap')
if __name__ == '__main__':
   main()