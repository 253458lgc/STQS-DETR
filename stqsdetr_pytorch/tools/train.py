"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning
    )

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="C:/use\MAIN\STQS-DETR-main/stqsdetr_pytorch/configs/stqsdetr/stqsdetr_regnet_6x_coco.yml")
    parser.add_argument('--resume', '-r', type=str,)
    parser.add_argument('--tuning', '-t', type=str,)
    parser.add_argument('--test-only', action='store_true',)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed',)
    args = parser.parse_args()

    main(args)


# def main(args, ) -> None:
#     '''main
#     '''
#     dist.init_distributed()
#     if args.seed is not None:
#         dist.set_seed(args.seed)
#
#     assert not all([args.tuning, args.resume]), \
#         'Only support from_scrach or resume or tuning at one time'
#
#     cfg = YAMLConfig(
#         args.config,
#         resume=args.resume,
#         use_amp=args.amp,
#         tuning=args.tuning
#     )
#
#     # 计算 FLOPs (如果需要)
#     if args.compute_flops:
#         try:
#             from fvcore.nn import FlopCountAnalysis, flop_count_table
#
#             model = cfg.model.to('cuda' if torch.cuda.is_available() else 'cpu')
#             model.eval()
#
#             # 创建输入张量
#             input_shape = [3, 640, 640]  # 默认输入大小
#             dummy_input = torch.rand(1, *input_shape).to(next(model.parameters()).device)
#
#             # 计算 FLOPs
#             flops = FlopCountAnalysis(model, dummy_input)
#
#             # 打印结果
#             print("\n" + "=" * 50)
#             print("模型 FLOPs 统计:")
#             print(flop_count_table(flops))
#             print(f"总 FLOPs: {flops.total() / 1e9:.2f} G")
#             print("=" * 50 + "\n")
#         except ImportError:
#             print("请安装 fvcore 库以计算 FLOPs: pip install fvcore")
#
#     solver = TASKS[cfg.yaml_cfg['task']](cfg)
#
#     if args.test_only:
#         solver.val()
#     else:
#         solver.fit()
#
#
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', '-c', type=str,
#                         default=r"C:/use\MAIN\STQS-DETR-main/stqsdetr_pytorch/configs/stqsdetr/stqsdetr_regnet_6x_coco.yml")
#     parser.add_argument('--resume', '-r', type=str, )
#     parser.add_argument('--tuning', '-t', type=str, )
#     parser.add_argument('--test-only', action='store_true', default=False, )
#     parser.add_argument('--amp', action='store_true', default=False, )
#     parser.add_argument('--seed', type=int, help='seed', )
#     parser.add_argument('--compute-flops', action='store_true', default="C:/use\MAIN\STQS-DETR-main/stqsdetr_pytorch/configs/stqsdetr/stqsdetr_regnet_6x_coco.yml",
#                         help='计算模型的 FLOPs')
#     args = parser.parse_args()
#
#     # 如果需要计算 FLOPs，导入必要的库
#     if args.compute_flops:
#         import torch
#
#     main(args)