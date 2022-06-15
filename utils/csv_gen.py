from cProfile import label
import os 
import pandas as pd
import time
import csv


def calss2_datacsv_gen(path):
 
    name_list = []
    label_list = []

    label = os.listdir(path)
    for i,label in enumerate(label):
        path_label = os.path.join(path, label)
        path_wsi = os.listdir(path_label)
        for j,name in enumerate(path_wsi):
            name_list.append(name)
            label_list.append(label)

    print(name_list)
    print(label_list)

    dataframe = pd.DataFrame({'pid':name_list,'label':label_list})
    dataframe.to_csv('train.csv',index=False,sep=',')

def csv_gen_new(slide_path):
            
    df = pd.read_csv(r'./data\RESULTS_DIRECTORY\process_list_autogen.csv') # 这个是上一步生成的文件
    ids1 = [i[:-4] for i in df.slide_id]
    ids2 = [i[:-3] for i in os.listdir(r'./data/RESULTS_DIRECTORY/patches/')]
    df['slide_id'] = ids1
    ids = df['slide_id'].isin(ids2)
    sum(ids)
    df.loc[ids].to_csv(r'./data\RESULTS_DIRECTORY\Step_2.csv',index=False)


    


    
    
    







if __name__ == '__main__':
    path = r'F:\Download\TCGA'
    csv_gen_new(path)
