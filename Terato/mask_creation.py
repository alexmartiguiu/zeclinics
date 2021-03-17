import ETL_lib as ETL
import sys
import os

if len(sys.argv) == 3:
    raw_data_path = sys.argv[1]
    output_path = sys.argv[2]
    if os.path.exists(raw_data_path):
        ETL.data_generation_pipeline(raw_data_path,output_path)
    else:
        print(raw_data_path,"Does not exist")
elif len(sys.argv) == 1:
    print("Using default parameters: raw_data_path = ../BAT1, output_path = ../Data")
    if os.path.exists('../BAT1'):
        ETL.data_generation_pipeline('../BAT1','../Data')
    else:
        print('./BAT1 Does not exist')
else:
    print("Usage: python3 mask_creation.py raw_data_path output_path")
