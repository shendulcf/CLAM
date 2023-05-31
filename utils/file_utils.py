import pickle
import h5py

# 用于将对象保存为 pickle 格式的文件。
# 它接收两个参数：filename 表示要保存的文件路径，save_object 表示要保存的对象。
# 函数内部使用 pickle.dump() 函数将对象保存到文件中
def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

# 用于从 pickle 文件中加载对象。
# 它接收一个参数 filename 表示要加载的文件路径。
# 函数内部使用 pickle.load() 函数从文件中加载对象，并返回加载的对象
def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file

# 用于将数据保存为h5格式的文件
def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path
'''
它接收四个参数：
output_path 表示要保存的文件路径，
asset_dict 表示要保存的数据字典，
attr_dict 表示要保存的数据属性字典（可选），
mode 表示文件打开模式，默认为 'a'。

函数内部使用 h5py.File() 创建 HDF5 文件对象，然后遍历 asset_dict 中的键值对，将数据保存到文件中。
如果数据对应的数据集不存在，则创建新的数据集并保存数据；
如果数据集已存在，则将新的数据追加到数据集的末尾。

如果提供了 attr_dict，则将属性字典中的属性保存到相应的数据集中。
最后，关闭文件并返回文件路径
'''