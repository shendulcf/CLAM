import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1 ----> 用于从给定的索引列表中顺序采样元素，无重复
class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

#2 ----> 用于对输入的批次数据进行整理和组装
def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]
#3 ----> 用于对特征数据的批次进行整理和组装
def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch]) ## 用于垂直（沿着行的方向）堆叠数组。它接受一个元组、列表或数组作为输入，并返回一个沿着垂直方向堆叠后的新数组
	return [img, coords]

#4 ----> 创建一个简单的数据加载器
def get_simple_loader(dataset, batch_size=1, num_workers=1):
	kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

#5 ----> 创建训练集或者验证机的数据加载器
def get_split_loader(split_dataset, training = False, testing = False, weighted = False):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL, **kwargs)	
			else:
				loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs )

	return loader

#6 ----> 用于计算用于平衡类别的样本权重
def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                

	return torch.DoubleTensor(weight)

# ----> 用于根据给定的优化器类型和参数创建优化器对象
def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	elif args.opt == 'adamW':
		optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'Radam':
		optimizer = optim.RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

# ----> 该函数的作用是打印网络模型的结构信息，并统计网络模型的总参数数量和可训练参数数量，用于了解网络模型的规模和参数情况
def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)

# ----> 用于生成训练验证测试集的划分
def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int) # samples patients or slide counts
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids) # setdiff1d(x,y) return in x notin y 
	'''
	首先，创建一个包含整数索引的数组indices，长度为samples，它代表所有样本的索引。如果提供了custom_test_ids（自定义测试集ID），
	则使用np.setdiff1d函数从indices中移除这些自定义测试集的索引
	'''
	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids
'''
参数：

cls_ids：一个包含类别索引的列表或数组，表示每个样本所属的类别。
val_num：一个包含验证集样本数量的列表或数组，表示每个类别在验证集中应有的样本数量。
test_num：一个包含测试集样本数量的列表或数组，表示每个类别在测试集中应有的样本数量。
samples：样本总数。
n_splits：划分的次数，默认为 5。
seed：随机种子，默认为 7。
label_frac：用于训练集的标签比例，默认为 1.0，表示全部样本。
custom_test_ids：自定义测试集样本索引的列表或数组，默认为 None。
返回值：
生成器对象，用于逐次产生训练集、验证集和测试集的划分。

函数内部首先根据 custom_test_ids 是否提供了自定义的测试集样本索引，来决定是否将其从整体样本中移除。

然后，根据 n_splits 参数指定的划分次数，进行以下操作：

初始化空列表 all_val_ids、all_test_ids 和 sampled_train_ids，用于存储每次划分的验证集、测试集和训练集样本索引。
对于每个类别 c：
使用 np.intersect1d 函数获取属于当前类别的样本索引。
从当前类别的样本索引中随机选择 val_num[c] 个样本作为验证集的索引，并将其添加到 all_val_ids 列表中。
在剩余的样本索引中随机选择 test_num[c] 个样本作为测试集的索引，并将其添加到 all_test_ids 列表中。
将剩余的样本索引添加到 sampled_train_ids 列表中。
如果 label_frac 不等于 1.0，则根据 label_frac 的比例选择部分剩余样本索引，并添加到 sampled_train_ids 列表中。
使用 yield 语句逐次生成训练集、验证集和测试集的样本索引划分
'''


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

# ----> 该函数的作用是计算预测结果与真实标签之间的错误率，用于评估模型在分类任务中的性能。错误率表示预测错误的样本占总样本数的比例，值越低表示模型性能越好
def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error
'''
函数内部首先将预测结果和真实标签转换为浮点数类型，并使用 .eq() 方法比较两个张量是否相等，得到一个布尔型的张量。
然后将布尔型张量转换为浮点数张量，并计算平均值。最后，通过 1 减去平均值得到错误率。
'''

# ----> 对模型的权重参数进行初始化，以便在训练过程中获得更好的收敛和性能
def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)
'''
函数内部通过遍历模型或模块中的参数，对不同类型的参数进行不同的初始化操作。
具体地，对于线性层（nn.Linear），使用 Xavier 初始化方法（nn.init.xavier_normal_）对权重进行初始化，并将偏置项置零（m.bias.data.zero_()）。
对于批归一化层（nn.BatchNorm1d），将权重参数设为常数 1，偏置项设为常数 0。
'''
