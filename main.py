import os

# os.environ['OMP_NUM_THREADS'] = '1'
import time
import itertools
import json
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import tqdm
import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.datasets import *
from src.environment.carla_gym_seq import docker_run_carla
from src.models.mvdet import MVDet
from src.models.mvcontrol import CamControl
from src.utils.logger import Logger
from src.utils.draw_curve import draw_curve
from src.utils.str2bool import str2bool
from src.trainer import PerspectiveTrainer


def main(args):
    torch.set_num_threads(1)

    # check if in debug mode
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
        torch.autograd.set_detect_anomaly(True)
    else:
        print('No sys.gettrace')
        is_debug = False

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # deterministic
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.backends.cudnn.benchmark = True

    # increase process niceness
    # os.nice(10)
    
    # id_ratio should be set to 0, if not using reID
    if not args.reID:
        args.id_ratio = 0
    else:
        # If in reID mode, the split ratio of reID test should be times by a multiplier
        split_ratio_lst = list(args.split_ratio)
        split_ratio_lst[-1] *= args.reid_testsize_multiplier
        args.split_ratio = tuple(split_ratio_lst)

    # dataset
    if args.dataset == 'carlax':
        carla_container = docker_run_carla(args.carla_gpu, args.carla_port) if not is_debug else None
        with open(f'./cfg/RL/{args.carla_cfg}.cfg', "r") as fp:
            dataset_config = json.load(fp)
        # reID mode, has to turn on motion :( !!!
        if args.reID:
            dataset_config['motion'] = True
        base = CarlaX(dataset_config, port=args.carla_port, tm_port=args.carla_tm_port, euler2vec=args.euler2vec)
        args.num_workers = 0
        args.batch_size = 1 if args.batch_size is None else args.batch_size

        if args.interactive:
            args.batch_size = 1
            args.augmentation = ''
            args.lr *= 0.1
            if not args.joint_training:
                args.base_lr_ratio = args.other_lr_ratio = 0
        if args.random_search:
            args.action_mapping = 'clip'
        X0, X1, Y0, Y1 = dataset_config['camera_range'][:4]
        # x0, x1, y0, y1 = dataset_config['spawn_area']
        args.div_xy_coef *= np.array([X1 - X0, Y1 - Y0]) / 20
    else:
        # args.augmentation += '+color'
        if args.dataset == 'wildtrack':
            base = Wildtrack(os.path.expanduser('~/Data/Wildtrack'))
        elif args.dataset == 'multiviewx':
            base = MultiviewX(os.path.expanduser('~/Data/MultiviewX'))
        else:
            raise Exception('must choose from [wildtrack, multiviewx]')
        args.batch_size = 1 if args.batch_size is None else args.batch_size
        args.interactive = False

    train_set = frameDataset(base, split='trainval', reID=args.reID, world_reduce=args.world_reduce,
                             world_kernel_size=args.world_kernel_size, img_kernel_size=args.img_kernel_size,
                             split_ratio=args.split_ratio, interactive=args.interactive, augmentation=args.augmentation,
                             tracking_scene_len=args.tracking_scene_len, seed=args.carla_seed)
    test_set = frameDataset(base, split='test', reID=args.reID, world_reduce=args.world_reduce,
                            world_kernel_size=args.world_kernel_size, img_kernel_size=args.img_kernel_size,
                            split_ratio=args.split_ratio, interactive=args.interactive,
                            tracking_scene_len=args.tracking_scene_len, seed=args.carla_seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)

    # logging
    RL_settings = (f'{"random" if args.random_search else "RL"}_fix_{args.reward}_{"J_" if args.joint_training else ""}'
                   f'{args.control_arch[0].upper()}_'
                   f'steps{args.ppo_steps}_b{args.rl_minibatch_size}_e{args.rl_update_epochs}_lr{args.control_lr}_'
                   f'{args.action_mapping}_ent{args.ent_coef}_stdinit{args.actstd_init}_'
                   f'{"det_" if args.rl_deterministic else ""}' if args.interactive else '')
    logdir = f'logs/{args.dataset}/{"DEBUG_" if is_debug else ""}' \
             f'{f"{args.carla_cfg}_{RL_settings}" if args.dataset == "carlax" else ""}' \
             f'TASK_{"reID" if args.reID else ""}_{args.aggregation}_e{args.epochs}_' \
             f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}' if not args.eval \
                else f'logs/{args.dataset}/EVAL_{f"{args.carla_cfg}" if args.dataset == "carlax" else ""}{args.resume}'
    os.makedirs(logdir, exist_ok=True)
    copy_tree('src', logdir + '/scripts/src')
    copy_tree('cfg', logdir + '/scripts/cfg')
    for script in os.listdir('.'):
        if script.split('.')[-1] == 'py':
            dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print(logdir)
    print('Settings:')
    print(vars(args))

    # model
    model = MVDet(train_set, args.arch, args.aggregation, args.use_bottleneck, args.hidden_dim, args.outfeat_dim,
                  args.dropout, check_visible=args.interactive).cuda()
    control_module = CamControl(train_set, args.hidden_dim, args.actstd_init, args.control_arch,
                                clip_action=args.action_mapping == 'clip', trig_yaw_coef=args.div_yaw_coef
                                ).cuda() if args.interactive else None

    # load checkpoint
    writer = None
    if args.interactive or args.resume or args.eval:
        load_dir = f'logs/{args.dataset}/{args.resume}' if args.resume else None
        if not os.path.exists(f'{load_dir}/model.pth'):
            # with open(f'logs/multiviewx/{args.arch}_None.txt', 'r') as fp:
            with open(f'logs/{args.dataset}/{args.arch}_{args.carla_cfg}{"_reID" if args.reID else ""}.txt', 'r') as fp:
                load_dir = fp.read()
        print(f'loading checkpoint: {load_dir}')
        pretrained_dict = torch.load(f'{load_dir}/model.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)  # the model may not have the reid head weights
        if args.interactive:
            if args.resume:
                control_module.load_state_dict(torch.load(f'logs/{args.dataset}/{args.resume}/control_module.pth'))
            if not is_debug:
                # tensorboard logging
                writer = SummaryWriter(logdir)
                writer.add_text("hyperparameters",
                                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|"
                                                                         for key, value in vars(args).items()])))

    if args.record_loss and not writer:
        # The case where we intentionally record the loss for each frame
        # Only initialize the writer if it was not initialized before
        writer = SummaryWriter(logdir)
        writer.add_text("hyperparameters",
                            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|"
                                                                        for key, value in vars(args).items()])))

    param_dicts = [{"params": [p for n, p in model.named_parameters()
                               if 'base' not in n and p.requires_grad],
                    "lr": args.lr * args.other_lr_ratio, },
                   {"params": [p for n, p in model.named_parameters()
                               if 'base' in n and p.requires_grad],
                    "lr": args.lr * args.base_lr_ratio, },
                   ]
    optimizer_model = optim.Adam(param_dicts, args.lr, weight_decay=args.weight_decay)

    if args.interactive and not args.random_search:
        param_dicts = [{"params": [p for n, p in control_module.named_parameters()
                                   if 'std' not in n and 'base' not in n and p.requires_grad],
                        "lr": args.control_lr, },
                       {"params": [p for n, p in control_module.named_parameters()
                                   if 'std' in n and p.requires_grad],
                        "lr": args.control_lr * args.std_lr_ratio, },
                       ]
        optimizer_agent = optim.Adam(param_dicts, args.control_lr, weight_decay=args.weight_decay)
    else:
        optimizer_agent = None

    def warmup_lr_scheduler(epoch, warmup_epochs=0.1 * args.epochs):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            # return (np.cos((epoch - warmup_epochs) / (args.epochs - warmup_epochs) * np.pi) + 1) / 2
            return 1 - (epoch - warmup_epochs) / (args.epochs - warmup_epochs + 1e-8)

    scheduler_model = None
    scheduler_agent = None

    trainer = PerspectiveTrainer(model, control_module, logdir, writer, args)

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    test_loss_s = []
    test_prec_s = []

    # Default value for epoch. When not training, epoch is not initialized, which
    # would lead to a bug.
    epoch = 1

    # learn
    use_best_action = args.random_search or args.reward == 'visibility'
    if not args.eval:
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss, train_prec = trainer.train(epoch, train_loader, (optimizer_model, optimizer_agent),
                                                   (scheduler_model, scheduler_agent))
            print('Testing...')
            test_loss, test_prec = trainer.test(test_loader, epoch,
                                                trainer.best_action if use_best_action else None, visualize=True)

            # draw & save
            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            train_prec_s.append(train_prec)
            test_loss_s.append(test_loss)
            test_prec_s.append(test_prec[0])
            draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, test_loss_s,
                       train_prec_s, test_prec_s)
            if args.interactive and not args.random_search:
                torch.save(control_module.state_dict(), os.path.join(logdir, 'control_module.pth'))
            if not args.interactive or args.joint_training:
                torch.save(model.state_dict(), os.path.join(logdir, 'model.pth'))

    print('Test loaded model...')
    print(logdir)
    if args.interactive:
        print(control_module.expand_mean_actions(test_set))
    trainer.test(test_loader, epoch + 1)
    if args.interactive:
        print('Test recorded best...')
        trainer.test(test_loader, epoch + 2, trainer.best_action)
    if args.dataset == 'carlax':
        base.env.close()
        if not is_debug:
            carla_container.stop()
    if args.interactive and not is_debug:
        writer.close()
    if not args.interactive and not args.eval:
        # We need to split the ckpt naming w, w/o reID setup
        with open(f'logs/{args.dataset}/{args.arch}_{args.carla_cfg}{"_reID" if args.reID else ""}.txt', 'w') as fp:
            fp.write(logdir)


if __name__ == '__main__':
    # common settings
    parser = argparse.ArgumentParser(description='view control for multiview detection')
    parser.add_argument('--eval', action='store_true', help='evaluation only')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--aggregation', type=str, default='max', choices=['mean', 'max'])
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack',
                        choices=['wildtrack', 'multiviewx', 'carlax'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=None, help='input batch size for training')
    parser.add_argument('--dropcam', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for task network')
    parser.add_argument('--base_lr_ratio', type=float, default=1.0)
    parser.add_argument('--other_lr_ratio', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--carla_seed', type=int, default=2023, help='random seed for CarlaX')
    parser.add_argument('--deterministic', type=str2bool, default=False)
    # MVcontrol settings
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--joint_training', type=str2bool, default=False)
    parser.add_argument('--carla_cfg', type=str, default=None)
    # parser.add_argument('--control_arch', default='transformer', choices=['conv', 'transformer'])
    parser.add_argument('--rl_deterministic', type=str2bool, default=True)
    parser.add_argument('--carla_port', type=int, default=2000)
    parser.add_argument('--carla_tm_port', type=int, default=8000)
    parser.add_argument('--carla_gpu', type=int, default=0)
    # RL arguments
    parser.add_argument('--random_search', action='store_true')
    parser.add_argument('--control_arch', type=str, default='encoder',
                        choices=['encoder', 'transformer', 'conv', 'linear'])
    parser.add_argument('--control_lr', type=float, default=1e-4, help='learning rate for MVcontrol')
    parser.add_argument('--euler2vec', type=str, default='yaw')
    parser.add_argument("--action_mapping", type=str, default='tanh', choices=['clip', 'tanh'],
                        help="the mapping of the action space")
    # https://arxiv.org/abs/2006.05990
    parser.add_argument('--actstd_init', type=float, default=0.5, help='initial value actor std')
    parser.add_argument("--reward", default='moda')
    parser.add_argument("--visibility_reduce", type=float, default=5)
    # https://www.reddit.com/r/reinforcementlearning/comments/n09ns2/explain_why_ppo_fails_at_this_very_simple_task/
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    # https://www.reddit.com/r/reinforcementlearning/comments/nr3weg/ppo_stuck_in_local_optimum/
    # https://github.com/Unity-Technologies/ml-agents/blob/c0889d5183411c393fa96387ff6cbb6ca5f1172a/config/ppo/Hallway.yaml
    parser.add_argument("--ppo_steps", type=int, default=512,
                        help="the number of steps to run in each environment per policy rollout, default: 2048")
    parser.add_argument("--rl_minibatch_size", type=int, default=128,
                        help="RL mini-batches, default: 64")
    parser.add_argument("--rl_update_epochs", type=int, default=10,
                        help="the K epochs to update the policy, default: 10")
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor, default: 0.99')
    parser.add_argument("--gae", type=str2bool, default=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--norm_adv", type=str2bool, default=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=str2bool, default=False,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent_coef", type=float, default=0.001,
                        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None,
                        help="the target KL divergence threshold")
    # additional loss/regularization
    parser.add_argument("--use_ppo", type=str2bool, default=True)
    parser.add_argument("--std_wait_epochs", type=int, default=0)
    parser.add_argument("--std_lr_ratio", type=float, default=10.0)
    parser.add_argument("--reg_decay_epochs", type=int, default=100)
    parser.add_argument("--reg_decay_factor", type=float, default=1.0)
    parser.add_argument("--steps_div_coef", type=float, default=0.1,
                        help="coefficient of chosen action diversity")
    parser.add_argument("--mu_div_coef", type=float, default=0.0,
                        help="coefficient of mean action diversity")
    parser.add_argument("--div_clamp", type=float, default=2.0,
                        help="clamp range of chosen action diversity")
    parser.add_argument("--div_xy_coef", type=float, default=1.0)
    parser.add_argument("--div_yaw_coef", type=float, default=0.5)
    parser.add_argument("--dir_coef", type=float, default=0.1,
                        help="coefficient of delta camera direction (compared to the location)")
    parser.add_argument("--cover_coef", type=float, default=0.0,
                        help="coefficient of the coverage")
    parser.add_argument("--cover_min_clamp", type=float, default=10,
                        help="clamp range of the coverage")
    parser.add_argument("--cover_max_clamp", type=float, default=200,
                        help="clamp range of the coverage")
    parser.add_argument("--autoencoder_coef", type=float, default=0.0,
                        help="coefficient of moda loss on x_feat")
    parser.add_argument("--autoencoder_detach", type=str2bool, default=False)
    # multiview detection specific settings
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--augmentation', type=str, default='affine')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--id_ratio', type=float, default=0.1)  # The default ratio is subject to change
    parser.add_argument('--cls_thres', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=0.0, help='ratio for per view loss')
    parser.add_argument('--use_mse', type=str2bool, default=False)
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--outfeat_dim', type=int, default=0)
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_kernel_size', type=int, default=10)
    # Additional settings for the JDETracker and tracking tast compability
    parser.add_argument('--split_ratio', nargs=3, type=float, default=(0.8, 0.1, 0.1),
                        help='train/val/test split ratio')
    parser.add_argument('--reid_testsize_multiplier', type=int, default=3,
                        help='number of times to multiply the number of test samples')
    parser.add_argument('--record_loss', action='store_true',
                        help='record loss for each frame')
    parser.add_argument('--tracking_scene_len', type=int, default=60,
                        help='number of frames for each tracking scene')
    parser.add_argument('--tracker_conf_thres', type=float, default=0.4,
                        help='object confidence threshold')
    parser.add_argument('--tracker_gating_threshold', type=float, default=1000,
                        help='gating threshold for motion fusion matching')
    # The argument below is used when we use reID classification head as reward function to train the control module
    parser.add_argument('--reid_acc_reward_factor', type=float, default=0.1)

    args = parser.parse_args()

    main(args)
