"""CLI：训练质心网络、独立跳跃与报告。"""
import argparse,json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description='G1关键节点质心协调与小跳')
    sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train'); t.add_argument('--output',default='runs/coordination/com_graph.pt'); t.add_argument('--samples',type=int,default=2500); t.add_argument('--epochs',type=int,default=120); t.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    j=sub.add_parser('jump'); j.add_argument('--output',default='runs/jump/manual'); j.add_argument('--checkpoint',default='runs/coordination/com_graph.pt'); j.add_argument('--viewer',action='store_true'); j.add_argument('--gif',action='store_true'); j.add_argument('--no-coordinator',action='store_true'); j.add_argument('--strength',type=float,default=1.)
    j.add_argument('--no-hold',action='store_true'); j.add_argument('--start-pose',nargs=4,type=float)
    args=p.parse_args(argv)
    if args.command=='train':
        import torch
        from fullbody.model import build_model
        from .network import train
        device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
        report=train(build_model(Path(args.output).parent/'model'),args.output,args.samples,args.epochs,device)
    else:
        from .jump import run
        report=run(args.output,args.viewer,None if args.no_coordinator else args.checkpoint,args.strength,args.gif,keep_open=not args.no_hold,start_pose=args.start_pose)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 2 if report.get('success') is False else 0


if __name__=='__main__': raise SystemExit(main())
