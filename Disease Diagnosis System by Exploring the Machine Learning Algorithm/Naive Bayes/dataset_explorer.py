import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def explore(dataset_df):
    disease=list()
    begin_index=list()
    end_index=list()
    class_prior_prob=list()
    class_count=list()

    no_rows=dataset_df['Disease'].count()
    y=""
    i=-1
    count=0

    for x in dataset_df['Disease']:
        i=i+1
        count=count+1
        if x!=y:
            begin_index.append(i)
            disease.append(x)
            if i!=0:
                end_index.append(i-1)
                class_count.append(count-1)
                class_prior_prob.append((count-1)/no_rows)
                count=1
        y=x

    end_index.append(i)
    class_count.append(count-1)
    class_prior_prob.append((count-1)/no_rows)

    disease_info={'Disease':disease,'begin_index':begin_index,'end_index':end_index,'class_count':class_count,'class_prior_prob':class_prior_prob}
    dataset_class_info=pd.DataFrame.from_dict(disease_info)

    return dataset_class_info
