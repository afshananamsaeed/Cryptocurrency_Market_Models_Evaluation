import os
os.chdir('/home/ubuntu/Masters_Thesis/scripts')
from data_modelling.nn_modelling import GetNNModelledOutput
import pandas as pd

if __name__ == "__main__":
    GetNNModelledOutput().run_modelling(dataset=1)
