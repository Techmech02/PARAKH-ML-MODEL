import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess_data(input_file='raw_mcqs.csv', output_file='preprocessed_mcqs.csv'):
    # Load the raw MCQs dataset
    df = pd.read_csv(input_file,on_bad_lines='skip')
    
    # Check for missing values and handle them if necessary (For now, just dropping rows with missing values)
    df.dropna(inplace=True)
    
    # Label Encoding Bloom's Taxonomy levels (Knowledge, Application, etc.)
    label_encoder = LabelEncoder()
    df['Bloom_level_encoded'] = label_encoder.fit_transform(df['Bloom_level'])
    
    # One-hot encoding the correct options (A, B, C, D)
    # Assuming correct_option column is like 'A', 'B', etc., we will create a new column for each option
    df['option_A'] = np.where(df['correct_option'] == 'A', 1, 0)
    df['option_B'] = np.where(df['correct_option'] == 'B', 1, 0)
    df['option_C'] = np.where(df['correct_option'] == 'C', 1, 0)
    df['option_D'] = np.where(df['correct_option'] == 'D', 1, 0)
    
    # Dropping the original correct_option column as it's now encoded in one-hot format
    df.drop(columns=['correct_option'], inplace=True)
    
    # Save the preprocessed dataset to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data()
