"""全身重定向/训练统一命令。"""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description='G1全身参考重定向与PPO模仿训练')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('retarget'); p.add_argument('pose'); p.add_argument('--output',default='motions/fullbody/reference.npz')
    p.add_argument('--root-mode',choices=('in_place','contact_grounded','auto'),default='contact_grounded')
    p.add_argument('--refine',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--keep-raw',action='store_true')
    p=sub.add_parser('train'); p.add_argument('reference'); p.add_argument('--output',default='runs/fullbody'); p.add_argument('--iterations',type=int,default=100); p.add_argument('--steps',type=int,default=256); p.add_argument('--seed',type=int,default=0); p.add_argument('--device',default='cpu',choices=['cpu','cuda']); p.add_argument('--resume')
    p.add_argument('--joint-assist',action=argparse.BooleanOptionalAction,default=True); p.add_argument('--joint-assist-gain',type=float,default=.18); p.add_argument('--airborne-reward',action=argparse.BooleanOptionalAction,default=True)
    for name in ('preview','evaluate'):
        p=sub.add_parser(name); p.add_argument('reference'); p.add_argument('--output',default='runs/fullbody/evaluation'); p.add_argument('--viewer',action='store_true')
        if name=='evaluate': p.add_argument('--checkpoint')
    p=sub.add_parser('amass-download'); p.add_argument('--repo',default='ember-lab-berkeley/AMASS_Retargeted_for_G1'); p.add_argument('--root'); p.add_argument('--mirror',default=None); p.add_argument('--workers',type=int,default=8)
    p=sub.add_parser('amass-list'); p.add_argument('--root')
    p=sub.add_parser('amass-export'); p.add_argument('--root'); p.add_argument('--output'); p.add_argument('--limit',type=int); p.add_argument('--overwrite',action='store_true'); p.add_argument('--contacts',action=argparse.BooleanOptionalAction,default=False)
    args=parser.parse_args(argv)
    if args.command=='retarget':
        from .reference import convert
        from .refine import refine
        output=Path(args.output)
        raw=output
        if args.refine:
            raw=output.with_name(output.stem+'_raw'+output.suffix)
        report=convert(args.pose,raw,root_mode=args.root_mode)
        if args.refine:
            refine(args.pose,raw,output)
            report.update(output=str(output),refined=True,raw_output=str(raw))
            if not args.keep_raw:
                raw.unlink(missing_ok=True)
        else:
            report['refined']=False
        print(json.dumps(report,ensure_ascii=False))
    elif args.command=='train':
        from .ppo import train
        train(args.reference,args.output,args.iterations,args.steps,args.seed,args.device,args.resume,
            joint_assist=args.joint_assist,joint_assist_gain=args.joint_assist_gain,airborne_reward=args.airborne_reward)
    elif args.command=='amass-download':
        from data_paths import AMASS
        from .amass import download, DEFAULT_MIRROR
        download(repo=args.repo,root=args.root or AMASS,mirror=args.mirror or DEFAULT_MIRROR,workers=args.workers)
    elif args.command=='amass-list':
        from data_paths import AMASS
        from .amass import inventory
        inventory(args.root or AMASS)
    elif args.command=='amass-export':
        from data_paths import AMASS, AMASS_MOTIONS
        from .amass import export
        export(root=args.root or AMASS,output=args.output or AMASS_MOTIONS,
               limit=args.limit,overwrite=args.overwrite,want_contacts=args.contacts)
    else:
        from .ppo import evaluate
        report=evaluate(args.reference,getattr(args,'checkpoint',None),args.output,args.viewer,args.command=='preview')
        if report['fallen']: return 2
    return 0


if __name__=='__main__':
    from viewer_camera import finish_process
    finish_process(main())
