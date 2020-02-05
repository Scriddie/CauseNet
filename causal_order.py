import numpy as np
import pandas as pd
import data_generating_process
from importlib import reload
reload(data_generating_process)

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

        # TODO: this part is still bugged
        mask = min_list <= min_of_max
        stage_members = tuple(df.columns[i:][mask])
        causal_stages.append(stage_members)

        del min_list[min_of_max_ind]
        del max_list[min_of_max_ind]
    
    return causal_stages


if __name__ == "__main__":
    df = data_generating_process.get_abc_df(
        10, data_generating_process.a_b_c)
    find_order(df)
