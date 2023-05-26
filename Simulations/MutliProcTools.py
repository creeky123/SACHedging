from multiprocessing import *
import numpy as np
import pandas as pd
import math

## multi processing based tools
## separate for multiprocessing support on windows

def open_scenario_files(file_path,queue):
    ## open pregenerated scenario files into the cache
    cache = []
    for path in file_path:
        cache.append(pd.read_csv(path))

    queue.put(cache)

def multi_proc_load_scens(file_paths):

    splitnum = 6
    split_paths = np.array_split(file_paths, splitnum)
    queue = Queue()

    for i in range(splitnum):
        p = Process(target=open_scenario_files, args=(split_paths[i],queue))
        p.start()

    out = []

    for i in range(splitnum):
        out.extend(queue.get())

    return out

def combine_df_columns(df_list, queue):
    columns = list(df_list[0].columns)[1:]
    time_df = pd.DataFrame(df_list[0][['# T']])
    dataframe_list = []
    for col in columns:
        dataframe_list.append(pd.concat([time_df.copy(),df_list[0][[col]]], axis=1))


    for idx in range(1,len(df_list)):
        this_df = df_list[idx]
        for xdx in range(0,len(columns)):
            df = dataframe_list[xdx]
            col = columns[xdx]
            df = pd.concat([df, this_df[[col]]],axis=1)
            dataframe_list[xdx] = df

    for df_1 in dataframe_list:
        columns = list(df_1.columns)
        for idx in range(1,len(columns)):
            columns[idx] = columns[idx] + str(idx)

    queue.put(dataframe_list)

def chunks(list_to_chunk, n):
    n = max(1, n)
    return (list_to_chunk[i:i+n] for i in range(0, len(list_to_chunk), n))

def multi_proc_combine_dfs(df_list):
    splitnum = 22
    split_df_list = chunks(df_list, math.floor(len(df_list)/splitnum))
    queue = Queue()

    for df in split_df_list:
        p = Process(target=combine_df_columns, args=(df,queue))
        p.start()

    out = []

    for i in range(splitnum):
        out.extend(queue.get())

    return out