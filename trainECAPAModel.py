'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time
from tools import *
from dataLoader import train_loader
from ECAPAModel import ECAPAModel
from torch.nn import DataParallel
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=200,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=128,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=0,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.002,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
# parser.add_argument('--train_list', type=str,   default="/home/zhangj21/projects/ECAPA/data/vox1_train_list.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
# parser.add_argument('--train_path', type=str,   default="/mnt/x3_home/zhangj21/attribution/GS/dev/wav",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
# parser.add_argument('--eval_list',  type=str,   default="/home/zhangj21/projects/ECAPA/data/test_list.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
# parser.add_argument('--eval_path',  type=str,   default="/mnt/x3_home/zhangj21/attribution/GS/test/wav",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
# parser.add_argument('--train_list1', type=str,   default="/mnt/x2/zhangj21/ECAPA-TDNN-main/voxceleb/1/doc/vox1_train_list.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
# parser.add_argument('--train_path1', type=str,   default="/home2/database/voxceleb/voxceleb1/dev/wav",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
# parser.add_argument('--eval_list1',  type=str,   default="/home/zhangj21/projects/ECAPA/test_list.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
# parser.add_argument('--eval_path1',  type=str,   default="/home2/database/voxceleb/voxceleb1/test/wav",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')




# parser.add_argument('--musan_path', type=str,   default="/mnt/x3_home/zhangj21/datasets/musan_split",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
# parser.add_argument('--rir_path',   type=str,   default="/mnt/x3_home/zhangj21/datasets/RIRS_NOISES/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case');
# parser.add_argument('--save_path',  type=str,   default="/home/zhangj21/ECAPA/exps/exp1",                                     help='Path to save the score.txt and models')
# parser.add_argument('--initial_model',  type=str,   default="",                                          help='Path of the initial_model')

# parser.add_argument('--train_list', type=str,   default="/home/zhangj21/projects/ECAPA-TDNN-main/voxceleb/2/doc/train_list.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
# parser.add_argument('--train_path', type=str,   default="/mnt/a3/cache/database/voxceleb/vox2/dev/aac",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
# parser.add_argument('--eval_list',  type=str,   default="/home/zhangj21/projects/ECAPA-TDNN-main/voxceleb/1/doc/veri_test2.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
# parser.add_argument('--eval_path',  type=str,   default="/mnt/database/sre/voxceleb/1/test/wav",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
# parser.add_argument('--musan_path', type=str,   default="/home/zhangj21/projects/ECAPA-TDNN-main/data/musan_split",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
# parser.add_argument('--rir_path',   type=str,   default="/home/zhangj21/projects/ECAPA-TDNN-main/data/RIRS_NOISES/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case');
# parser.add_argument('--save_path',  type=str,   default="/home/zhangj21/projects/ECAPA-TDNN-main/exps/exp1",                                     help='Path to save the score.txt and models')
# parser.add_argument('--initial_model',  type=str,   default="",                                          help='Path of the initial_model')
parser.add_argument('--train_list', type=str,   default="/home/data/voxceleb/voxceleb2/train_list.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--train_path', type=str,   default="/home/mnt/a3/database/sre/voxceleb/vox2_wav/dev/aac",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
parser.add_argument('--eval_list',  type=str,   default="/home/data/voxceleb/voxceleb1/veri_test2.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--eval_path',  type=str,   default="/home/mnt/a3/database/sre/voxceleb/1/test/wav",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
parser.add_argument('--musan_path', type=str,   default="/home/mnt/a5/zhangj21/projects1/xiugaiECAPA-TDNN-main/data/musan_split",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--rir_path',   type=str,   default="/home/mnt/a5/zhangj21/projects1/xiugaiECAPA-TDNN-main/data/RIRS_NOISES/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case');
parser.add_argument('--save_path',  type=str,   default="/home/code/ECAPA/exps/exp1",                                     help='Path to save the score.txt and models')
parser.add_argument('--initial_model',  type=str,   default="",                                          help='Path of the initial_model')


## Model and Loss settings
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=5994,   help='Number of speakers')

## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')
parser.add_argument('--local_rank', default=-1, type=int,          help='node rank for distributed training')
dist.init_process_group(backend='nccl')
#device = torch.device("cuda", args.local_rank)


## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)
torch.cuda.set_device(args.local_rank)

## Define the data loader
torch.multiprocessing.set_start_method('spawn', force=True)
trainloader = train_loader(**vars(args))
train_sampler = torch.utils.data.distributed.DistributedSampler(trainloader)
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size ,sampler=train_sampler, shuffle = False, num_workers = args.n_cpu, drop_last = True)

## Search for the exist models
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()
device = torch.device("cuda")
## Only do evaluation, the initial_model is necessary
if args.eval == True:
	s = ECAPAModel(**vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)
	EER, minDCF = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
	print("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
	quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
	print("Model %s loaded from previous state!"%args.initial_model)
	s = ECAPAModel(**vars(args))
	s.load_parameters(args.initial_model)
	epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
	print("Model %s loaded from previous state!"%modelfiles[-1])
	epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
	s = ECAPAModel(**vars(args))
	s.load_parameters(modelfiles[-1])
## Otherwise, system will train from scratch
else:
	epoch = 1
	s = ECAPAModel(**vars(args))

EERs = []
score_file = open(args.score_save_path, "a+")

while(1):
	## Training for one epoch
	loss, lr, acc = s.train_network(epoch = epoch, loader = trainLoader)

	## Evaluation every [test_step] epochs
	if epoch % args.test_step == 0:
		s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
		EERs.append(s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)[0])
		print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, acc, EERs[-1], min(EERs)))
		score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, lr, loss, acc, EERs[-1], min(EERs)))
		score_file.flush()

	if epoch >= args.max_epoch:
		quit()

	epoch += 1
