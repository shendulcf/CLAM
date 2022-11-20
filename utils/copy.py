import sys
import os
import shutil

# find all folders in src_base_dir 
# go to one of it
# move files in the folder to dst 

src_base_dir = r"/home/sci/Disk2/TCGA-BRCA/WSI"
dst = r"/home/sci/Disk2/tcga_brca(1)"
for folder in os.listdir(src_base_dir):
    print(folder)
    filePath = src_base_dir + '\\'  + folder
    filePath2 = os.path.join(src_base_dir, folder)
    print(filePath)
    # if filePath in dst:continue
    for file in os.listdir(filePath2):
        src = filePath + '\\' + file
        src2 = os.path.join(filePath2, file)
        print(src2)
        # shutil.move(src, dst)
        shutil.copy(src2, dst)

# all_slides = glob.glob(join(path_base, '*/*.svs'))