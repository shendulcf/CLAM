from cProfile import label
import os 
import pandas as pd
import time
import csv
from pathlib import Path


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

def csv_gen_test(path,result_dir):

    slide_list = os.listdir(path)
    slide_list.sort()
    case_name,slide_name,label_name = [],[],[]
    # print(slide_list)
    for slide in slide_list:
        slide_n, suffix = os.path.splitext(slide)
        if slide[13] == '1':
            label = "normal_tissue"
        else:
            label = "tumor_tissue"
        case_n = slide_n[:12]
        case_name.append(case_n)
        slide_name.append(slide_n)
        label_name.append(label)
    
    data = {"case_id":case_name,
            "slide_id":slide_name,
            "label":label_name
    }
    frame = pd.DataFrame(data)
    frame.to_csv(result_dir)


def csv_gen_step1(csv_dir,result_dir):
            
    df = pd.read_csv(csv_dir) # 这个是上一步生成的csv文件
    ids1 = [i[:-4] for i in df.slide_id]
    ids2 = [i[:-3] for i in os.listdir(r'data_tcga_crc/RESULTS_DIRECTORY/patches')]
    df['slide_id'] = ids1
    ids = df['slide_id'].isin(ids2)
    sum(ids)
    df.loc[ids].to_csv(result_dir,index=False)  # data2/RESULTS_DIRECTORY/step_2.csv

def csv_gen_step2(csv_dir,result_dir):

    df = pd.read_csv(csv_dir)
    df = df[['Case_ID','Slide_ID','Specimen_Type']]
    ids1 = [i for i in df.Slide_ID]
    ids2 = [i[:-3] for i in os.listdir('toy_test/patches/')]
    ids = df['Slide_ID'].isin(ids2)
    sum(ids)
    df = df.loc[ids]
    df.columns = ['case_id','slide_id','label']
    df.to_csv(result_dir,index=False)



    
    

if __name__ == '__main__':
    path = r'data_tcga_crc/DATA_DIRECTORY'
    path2 = r'/home/sci/Disk2/tcga_crc/DATA_DIRECTORY'
    fpath = r'/home/sci/Disk2/tcga_crc/FEATURES_DIRECTORY'

    csv_dir = r'data_tcga_crc/RESULTS_DIRECTORY/process_list_autogen.csv'
    result_dir = r'data_tcga_crc/RESULTS_DIRECTORY/step_2.csv'
    result2_dir = r'data_tcga_crc/RESULTS_DIRECTORY/step_3.csv'
    print(os.listdir(path))
    
    print(os.getcwd())
    # csv_gen_step1(csv_dir,result_dir)
    csv_gen_test(path2,result2_dir)
    # csv_gen_step2(csv)
    
    