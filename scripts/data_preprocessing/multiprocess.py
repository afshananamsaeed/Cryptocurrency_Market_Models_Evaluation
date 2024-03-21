import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import get_context

class MultiProcessText:
    def __init__(self):
        pass
    
    @staticmethod       
    def parallelize_dataframe(df, func):
        num_processes = 32  #mp.cpu_count() # or consider taking 16/32
        # df_split = np.array_split(df, num_processes)
        df_split = np.array_split(df, 5*num_processes)
        with get_context("spawn").Pool(num_processes) as p:
            try:
                results = list(tqdm(p.imap(func, df_split), total = len(df_split)))
                p.close()
                p.join()
                df = pd.concat(results)
            except KeyboardInterrupt:
                p.terminate()
            except:
                p.terminate()
        return df