import torch
import os
import shutil

def LoadModel(Path,Model,Optimizer,start_epoch,best_prec):
	if os.path.isfile(Path):
		checkpoint = torch.load(Path)
		start_epoch = checkpoint['epoch']
		best_prec = checkpoint['best_prec1']
		Model.load_state_dict(checkpoint['state_dict'])
		Optimizer.load_state_dict(checkpoint['optimizer'])
		print("=> loaded checkpoint '{}' (epoch {})".format(Path, checkpoint['epoch']))
	else:
		print("=> no checkpoint found at '{}'".format(Path))
	return start_epoch,best_prec,Model,Optimizer


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')