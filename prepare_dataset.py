from urllib.request import urlretrieve
import os.path
import random 


from scipy.io.arff import loadarff
import pandas as pd



def load_heart_statlog(filename):
    data, metadata = loadarff(filename)
    df = pd.DataFrame(data)
    column_types = {'age': int, 
                    'sex': [0,1], 
                    'chest': float, 
                    'resting_blood_pressure': float, 
                    'serum_cholestoral': float, 
                    'fasting_blood_sugar': [0,1], 
                    'resting_electrocardiographic_results': [0,1,2], 
                    'maximum_heart_rate_achieved': float, 
                    'exercise_induced_angina': [0,1], 
                    'oldpeak': float, 
                    'slope': int, 
                    'number_of_major_vessels': float, 
                    'thal': [3,6,7], 
                    'class': [b'present', b'absent']}
    feature_columns = ['age', 'sex', 'chest', 'resting_blood_pressure', 'serum_cholestoral', 'fasting_blood_sugar', 'resting_electrocardiographic_results', 'maximum_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak', 'slope', 'number_of_major_vessels', 'thal', ]
    label = df[['class']].map(lambda x: 1 if x == b'present' else 0)  # Note the double bracket, this keeps the column as a DataFrame
    data = df[feature_columns]
    categorical_features = [feature for feature in feature_columns if column_types[feature] not in (int, float)]
    data = data.astype({col: 'category' for col in categorical_features})
    #dataset = lgb.Dataset(data, label=label, free_raw_data=False, categorical_feature=categorical_features)
    return data, label


def main():
    dataset_url = "https://www.openml.org/data/download/6358/BNG_heart-statlog.arff"
    dataset_file = "BNG_heart-statlog.arff"
    output_file_full = "BNG_heart-statlog.feather"
    output_file_subsample = "BNG_heart-statlog_subsample.feather"
    
    subsample = 0.2
    
    if not os.path.exists(output_file_full):
        if not os.path.exists(dataset_file):
            urlretrieve(dataset_url, dataset_file)
        data, label = load_heart_statlog(dataset_file)
        df = pd.concat([data, label], axis=1)
        df.to_feather(output_file_full)
    
    if not os.path.exists(output_file_subsample):
        df = pd.read_feather(output_file_full)
        n = len(df)
        k = int(n*subsample)
        rng = random.Random(1729)
        indices = rng.sample(range(n), k=k)
        subsampled_df = df.iloc[indices].reset_index()
        
        subsampled_df.to_feather(output_file_subsample)
        
    
if __name__ == '__main__':
    main()