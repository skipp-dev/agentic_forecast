
import pandas as pd
import numpy as np

def test_dict_conversion():
    # Create a sample dataframe
    df = pd.DataFrame({
        'close': [100, 101, 102],
        'returns': [0.01, 0.01, 0.01]
    }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
    
    print("Original DataFrame:")
    print(df)
    
    # Simulate agent_nodes.py logic
    df.index = df.index.astype(str)
    data_dict = df.to_dict('index')
    
    print("\nDict representation:")
    print(data_dict)
    
    # Simulate execution_nodes.py logic
    restored_df = pd.DataFrame.from_dict(data_dict, orient='index')
    restored_df.index = pd.to_datetime(restored_df.index)
    
    print("\nRestored DataFrame:")
    print(restored_df)
    print("\nColumns:", restored_df.columns.tolist())
    
    if 'close' not in restored_df.columns:
        print("\nFAIL: 'close' column missing!")
    else:
        print("\nSUCCESS: 'close' column present.")

if __name__ == "__main__":
    test_dict_conversion()
