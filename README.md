This is a collection of models that use a hospital patient's historical medical
records, vital/lab measuremnts and socioeconomic information to predict the 
probability of them being readmitted to the hospital within 30 days.  

"patient_cohort_construction.py": script to assemble a dataframe containing the 
   historical data and observations (which we use as features for the model).

"readmissions_dnn.py": a TFLearn deep neural net model
