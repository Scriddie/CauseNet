import numpy as np
import pandas as pd

def find_order(df):
    min_list = []
    max_list = []

    for n in df.columns:
        time_col = f"{n}_time"
        x = pd.DataFrame(df[n].to_list(), columns=[n, time_col])
        col_min, col_max = min(x[time_col]), max(x[time_col])
        min_list.append(col_min)
        max_list.append(col_max)

    causal_stages = []
    for i, col_name in enumerate(df.columns):
        min_of_max = np.min(max_list)
        min_of_max_ind = np.argmin(max_list)

        mask = min_list <= min_of_max
        stage_members = tuple(df.columns[i:][mask])
        causal_stages.append(stage_members)

        del min_list[min_of_max_ind]
        del max_list[min_of_max_ind]


