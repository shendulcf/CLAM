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

def csv_gen_test(path):

    slide_list = os.listdir(path)
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
    frame.to_csv('data/RESULTS_DIRECTORY/step_3.csv')


def csv_gen_step1():
            
    df = pd.read_csv(r'data/RESULTS_DIRECTORY/process_list_autogen.csv') # 这个是上一步生成的文件
    ids1 = [i[:-4] for i in df.slide_id]
    ids2 = [i[:-3] for i in os.listdir(r'data/RESULTS_DIRECTORY/patches')]
    df['slide_id'] = ids1
    ids = df['slide_id'].isin(ids2)
    sum(ids)
    df.loc[ids].to_csv('data/RESULTS_DIRECTORY/Step_2.csv',index=False)

def csv_gen_step2():

    df = pd.read_csv('/media/yuansh/14THHD/CLAM/DataSet/cohort.csv')
    df = df[['Case_ID','Slide_ID','Specimen_Type']]
    ids1 = [i for i in df.Slide_ID]
    ids2 = [i[:-3] for i in os.listdir('toy_test/patches/')]
    ids = df['Slide_ID'].isin(ids2)
    sum(ids)
    df = df.loc[ids]
    df.columns = ['case_id','slide_id','label']
    df.to_csv('/media/yuansh/14THHD/CLAM/Step_3.csv',index=False)



    
    
    

if __name__ == '__main__':
    path = r'/home/sci/PycharmProjects/chaofan/projects/Datasets/tcga_test'
    
    print(os.getcwd())
    # csv_gen_step1()
    csv_gen_test(path)
