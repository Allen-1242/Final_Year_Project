import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from dataset_explorer import explore

dataset_df = pd.read_csv("dataset.csv")

print(dataset_df)

training_data = np.array(dataset_df['Symptoms'].values.astype('U'))#training data(read as unicode values)

print(training_data)

cv = CountVectorizer()

cv.fit(training_data)

training_data_features = cv.get_feature_names()#feature vector for training data

print("\n******************Feature vector of the training data*****************\n")
print(np.array(training_data_features))
print("\n**********************************************************************\n")

training_data_dtm = cv.transform(training_data)#document term matrix for training data

print("\n****************Document term matrix for training data****************\n")
print(training_data_dtm)
print("\n**********************************************************************\n")

features_ft = pd.DataFrame(training_data_dtm.toarray(),columns=training_data_features)#frequency table for features

print("\n***********************Frequency table for features*******************\n")
print(features_ft)
print("\n**********************************************************************\n")

print("\n********Frequency of occourence of each disease in the dataset********\n")
print(dataset_df.Disease.value_counts())
print("\n**********************************************************************\n")

print("\n**************************Dataset exploration*************************\n")
disease_info=explore(dataset_df)
disease_info.set_index('Disease',inplace=True)
print(disease_info)
print("\n**********************************************************************\n")

print("\n************************Symptom Frequency table***********************\n")
symptom_ft=pd.DataFrame(dataset_df['Symptoms'].value_counts())
symptom_ft.columns=['frequency']
print(symptom_ft)
print("\n**********************************************************************\n")

def predict(test_features):
    
    features_prior_prob=1

    print("\n********Calculating prior probability of features********\n")
    for x in test_features:
        print("Prior Probability of ",x,": ",(symptom_ft.at[x,'frequency']/len(disease_info.index)),sep='')
        features_prior_prob=features_prior_prob*(symptom_ft.at[x,'frequency']/len(disease_info.index));
    print("\nFeatures_prior_prob: ",features_prior_prob,sep='')
    print("\n*********************************************************\n")
    
    likelihood=1
    predicted_class_prob=0
    predicted_class=""

    print("\n********Calculating posterior probability of class********\n")
    for x in disease_info.index:
        likelihood=1
        for y in test_features:
            class_symptom_ft=pd.DataFrame(dataset_df[disease_info.at[x,'begin_index']:(disease_info.at[x,'end_index']+1)]['Symptoms'].value_counts())
            class_symptom_ft.columns=['frequency']
            if(y in class_symptom_ft.index):
                likelihood=likelihood*(class_symptom_ft.at[y,'frequency']/len(class_symptom_ft.index))
            else:
                likelihood=0
                break
                
        class_post_prob=(likelihood*disease_info.at[x,'class_prior_prob'])/features_prior_prob
        print("Posterior proability of ",x,": ",class_post_prob,sep='')
        if(class_post_prob > predicted_class_prob):
            predicted_class_prob=class_post_prob
            predicted_class=x
    print("\n**********************************************************\n")

    print("\n************************Diaganosis************************\n")
    if(predicted_class_prob!=0):
        print("Data suggests that there is a %.2f"%(predicted_class_prob*100),"% chance of \'",predicted_class,"\'",sep='')
    else:
        print("Anamoly detected in input symptoms! No prediction possible")
    print("\n**********************************************************\n")
    
predict(["alcoholic withdrawal symptoms","unconscious state","formication","withdraw","todd paralysis","tremor"])

