This repository includes data_cleaning.py file for Data cleaning. Orginal data set can be accessed from here: https://www.kaggle.com/datasets/mohameds10960/car-fuel-and-emissions-2000-2013
Second file is model_building, for that we will use df_encoded.csv file created from step 1 and saved (available in repository as well) and save the Machine learning model (also avaiable in repository).
Third file is model_io.py file to load model saved in step 2.
Fourth file is api.py for creating geadio interface and Fast Api. to run Fast Api you will have to renew ngrok verification token. 
fifth file is app.py which is a code file to deploy Fast Api on hugging face using interface from Graio.
Gradio interface deployed at https://nehal61-fast-api.hf.space and can be accessed for using model.
