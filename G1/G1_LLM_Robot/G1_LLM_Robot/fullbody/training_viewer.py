"""持续渲染窗口：状态栏 + 最新策略独立物理回放，关闭窗口不停止训练。"""
from pathlib import Path
import sys
import time
import traceback
import tkinter as tk
from datetime import datetime
from .jobs import read_json,atomic_json
from viewer_camera import TRAINING_DISTANCE,apply_isometric_camera,training_window_size
from proactive import format_duration

# 修复：load_checkpoint 原先在 tick() 内部 import，但同一函数作用域内更早的分支
# 也会用到它，Python 因此把它当作局部变量，触发 UnboundLocalError。
# 提到模块顶层后，函数内不再有 import 语句，名字解析走全局作用域。
try:
    from runtime_io import load_checkpoint
except Exception:  # 允许在 runtime_io 不可用时仍能打开窗口（仅显示状态栏）
    load_checkpoint=None


def main(folder):
    folder=Path(folder)
    window=tk.Tk(); window.title('G1 视频训练实时监视 — '+folder.name)
    screen_w=window.winfo_screenwidth(); screen_h=window.winfo_screenheight()
    win_w,win_h=training_window_size(screen_w,screen_h)
    window.geometry(f'{win_w}x{win_h}+{max(0,(screen_w-win_w)//2)}+{max(0,(screen_h-win_h)//3)}')
    window.minsize(min(1024,win_w),min(640,win_h))
    title=tk.StringVar(value='等待后台训练进程…')
    tk.Label(window,textvariable=title,font=('Microsoft YaHei',11),justify='left',wraplength=760).pack(fill='x',padx=10,pady=8)
    image_label=tk.Label(window,text='正在准备视频和参考动作',bg='#172331',fg='white')
    image_label.pack(fill='both',expand=True)
    tk.Label(window,text='显示最近保存策略的独立物理回放；每次片段结束或摔倒重置。关闭此窗口不会停止训练。').pack(pady=5)
    state={'env':None,'policy':None,'renderer':None,'renderer_size':None,'stamp':None,'obs':None,'frame':0,'last_check':0,'iteration':None,'resize_job':None,
        'render_scale':0.85,'render_ms':0.0,'quality_counter':0,'photo':None,'last_error':None,'closed':False}
    def desired_render_size():
        """按图像区域的实际像素尺寸选择离屏渲染分辨率。"""
        window.update_idletasks()
        width=image_label.winfo_width(); height=image_label.winfo_height()
        if width<320 or height<240:
            width=max(960,min(1600,window.winfo_width()-40))
            height=max(540,min(900,window.winfo_height()-140))
        scale=min(1.0,max(.5,float(state['render_scale'])))
        width=max(640,min(1600,int(width*scale))); height=max(360,min(900,int(height*scale)))
        width-=width%2; height-=height%2
        return height,width
    def ensure_renderer(force=False):
        env=state['env']
        if env is None: return
        import mujoco
        size=desired_render_size()
        if not force and state['renderer'] is not None and state['renderer_size']==size: return
        if state['renderer'] is not None:
            try: state['renderer'].close()
            except Exception: pass
        state['photo']=None
        try: image_label.configure(image='')
        except Exception: pass
        env.model.vis.global_.offwidth=max(env.model.vis.global_.offwidth,size[1])
        env.model.vis.global_.offheight=max(env.model.vis.global_.offheight,size[0])
        state['renderer']=mujoco.Renderer(env.model,height=size[0],width=size[1])
        state['renderer_size']=size
    def apply_resize():
        state['resize_job']=None
        try: ensure_renderer()
        except Exception: pass
    def schedule_resize(event=None):
        if event is not None and event.widget is not window: return
        if state['resize_job'] is not None:
            try: window.after_cancel(state['resize_job'])
            except Exception: pass
        state['resize_job']=window.after(300,apply_resize)
    window.bind('<Configure>',schedule_resize)
    def close():
        if state['closed']:
            return
        state['closed']=True
        if state['resize_job'] is not None:
            try: window.after_cancel(state['resize_job'])
            except Exception: pass
        if state['renderer'] is not None:
            try: state['renderer'].close()
            except Exception: pass
        if state['env'] is not None:
            try: state['env'].close()
            except Exception: pass
        atomic_json(folder/'viewer_status.json',{'state':'closed'})
        try: window.destroy()
        except Exception: pass
    window.protocol('WM_DELETE_WINDOW',close)
    atomic_json(folder/'viewer_status.json',{'state':'waiting','message':'窗口已打开，等待参考动作'})
    def tick():
        if state['closed']: return
        if not window.winfo_viewable():
            window.after(250,tick); return
        if (folder/'CLOSE_VIEWER').exists(): close(); return
        try:
            status=read_json(folder/'status.json',{})
            metrics=status.get('metrics',{})
            if status.get('duration_text') is not None:
                elapsed_text=status.get('duration_text')
            elif status.get('started_at'):
                try:
                    started=datetime.fromisoformat(status['started_at'])
                    if started.tzinfo is None: started=started.astimezone()
                    elapsed_text=format_duration(
                        (datetime.now().astimezone()-started).total_seconds())
                except (TypeError,ValueError):
                    elapsed_text=status.get('elapsed_text','—')
            else:
                elapsed_text=status.get('elapsed_text','—')
            title.set(f"阶段: {status.get('state','queued')} | 设备: {status.get('device','检测中')} | 已用: {elapsed_text} | 预计剩余: {status.get('eta_text','—')}\n"
                f"轮次: {status.get('iteration',0)}/{status.get('total_iterations','?')}\n"
                f"{status.get('message','')}\n平均奖励: {metrics.get('mean_reward','—')} | 画面权重轮次: {state['iteration'] or '平衡基线'}")
            reference=folder/'reference.npz'
            if reference.is_file() and state['env'] is None:
                import torch
                import mujoco
                from .balanced import BalancedEnv
                from .ppo import ActorCritic
                torch.set_num_threads(1)
                env=BalancedEnv(reference,folder/'viewer_model')
                config=read_json(folder/'config.json',{}) or {}
                env.video_leg_weight=config.get('leg_weight',.15)
                env.airborne_reward=bool(config.get('airborne_reward',True))
                state['env']=env; state['policy']=ActorCritic().eval()
                torch.nn.init.zeros_(state['policy'].actor[-1].weight); torch.nn.init.zeros_(state['policy'].actor[-1].bias)
                camera=mujoco.MjvCamera()
                apply_isometric_camera(camera,distance=TRAINING_DISTANCE,lookat=(0,0,.65))
                state['camera']=camera
                state['obs'],_=env.reset(options={'phase':0})
                ensure_renderer()
                image_label.configure(text='')
            env=state['env']
            if env is not None:
                import torch
                import numpy as np
                from PIL import Image,ImageTk
                if state['renderer'] is None:
                    ensure_renderer()
                if time.monotonic()-state['last_check']>2:
                    state['last_check']=time.monotonic()
                    checkpoint=folder/'training/latest.pt'
                    if load_checkpoint is not None and checkpoint.is_file() and checkpoint.stat().st_mtime_ns!=state['stamp']:
                        try:
                            payload=load_checkpoint(checkpoint,map_location='cpu',weights_only=True)
                            env.video_leg_weight=payload.get('leg_weight',.15)
                            env.video_arm_weight=payload.get('arm_weight',1.)
                            env.airborne_reward=bool(payload.get('airborne_reward',True))
                            from coordination.video import apply_motion, restore_controller
                            apply_motion(env,payload)
                            if env.video_controller_config!=payload.get('video_controller',{}):
                                restore_controller(env,payload)
                                state['obs'],_=env.reset(options={'phase':0})
                            state['policy']=ActorCritic.from_checkpoint(payload,eval_mode=True); state['iteration']=payload['iteration']
                            state['stamp']=checkpoint.stat().st_mtime_ns
                        except Exception:
                            # 训练进程可能正在原子替换权重；读失败不应终止整个渲染窗口。
                            pass
                for _ in range(3):
                    with torch.no_grad(): action=state['policy'].mean(torch.as_tensor(state['obs']).unsqueeze(0))[0].numpy()
                    obs,reward,fallen,ended,info=env.step(action)
                    state['obs']=obs
                    if fallen or ended: state['obs'],_=env.reset(options={'phase':0})
                render_start=time.perf_counter()
                try:
                    state['camera'].lookat[:]=[env.data.qpos[0],env.data.qpos[1],.65]
                    state['renderer'].update_scene(env.data,camera=state['camera'])
                    # 拷贝内部复用缓冲，避免后续渲染覆盖当前画面。
                    pixels=state['renderer'].render().copy()
                except Exception:
                    if state['renderer'] is not None:
                        try: state['renderer'].close()
                        except Exception: pass
                    state['renderer']=None; state['renderer_size']=None; state['photo']=None
                    raise
                image=Image.fromarray(pixels)
                if state['frame']%120==0:
                    preview=image.copy(); preview.thumbnail((960,540))
                    preview.save(folder/'viewer_preview.png')
                photo=state['photo']
                if photo is None or photo.width()!=image.width or photo.height()!=image.height:
                    photo=ImageTk.PhotoImage(image); state['photo']=photo
                    image_label.configure(image=photo)
                else:
                    try: photo.paste(image)
                    except Exception:
                        photo=ImageTk.PhotoImage(image); state['photo']=photo
                        image_label.configure(image=photo)
                image_label.image=photo
                elapsed_ms=(time.perf_counter()-render_start)*1000
                state['render_ms']=(elapsed_ms if not state['render_ms'] else .85*state['render_ms']+.15*elapsed_ms)
                state['quality_counter']+=1
                # 质量自适应使用长周期与宽死区，避免频繁销毁/重建 OpenGL
                # 渲染器造成画面闪烁或驱动报错。
                if state['quality_counter']>=120:
                    state['quality_counter']=0
                    if state['render_ms']>85 and state['render_scale']>.5:
                        state['render_scale']=max(.5,state['render_scale']*.80); ensure_renderer(force=True)
                    elif state['render_ms']<18 and state['render_scale']<1.0:
                        state['render_scale']=min(1.0,state['render_scale']*1.10); ensure_renderer(force=True)
                if state['frame']%30==0:
                    atomic_json(folder/'viewer_status.json',{'state':'rendering','checkpoint_iteration':state['iteration'],
                        'last_rollout_fallen':fallen,'simulation_phase':env.phase,
                        'render_ms':round(state['render_ms'],1),'render_scale':round(state['render_scale'],3)})
                state['frame']+=1
        except tk.TclError:
            # 窗口已关闭时不再写状态或安排回调。
            return
        except Exception as exc:
            title.set('渲染错误：'+str(exc)+'；训练在独立进程中继续。')
            if state['last_error']!=str(exc):
                state['last_error']=str(exc)
                atomic_json(folder/'viewer_status.json',{'state':'failed','error':str(exc),'traceback':traceback.format_exc()})
            if not state['closed']:
                window.after(1000,tick)
            return
        if not state['closed']:
            window.after(60,tick)
    window.after(100,tick)
    window.mainloop()


if __name__=='__main__':
    try: main(sys.argv[1])
    except Exception as exc:
        atomic_json(Path(sys.argv[1])/'viewer_status.json',{'state':'failed','error':str(exc)})
        raise
