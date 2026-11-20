"""工具结果的控制台展示；原始结果仍完整回传模型。"""
import json
import math

TRAINING_TOOLS={'robot_jump_handoff','robot_jump_close_window','robot_coordination_start','robot_coordination_status','robot_learn_action','robot_search_action_videos','robot_amass_train','robot_multiview_train','robot_train_video','robot_training_status','robot_proactive_report','robot_training_stop','robot_training_viewer','robot_docker_train','robot_docker_status','robot_docker_summary','robot_docker_export_video','robot_docker_stop','robot_export_gif','robot_gif_status'}
STATES={'handed_off':'已交接到行走仿真','ready_for_handoff':'落地完成，等待交接','coordinator_preparing':'准备质心协调网络','searching':'搜索候选视频','downloading':'下载候选视频','screening':'全身动作筛选','preparing':'准备学习数据','queued':'已排队','extracting':'提取骨架','retargeting':'全身重定向','training':'训练中','evaluating':'独立评估','completed':'任务完成','failed':'失败','cancelled':'已停止','rendering':'渲染中','waiting':'等待参考','closed':'窗口已关闭'}


def format_result(name,result):
    lines=[f'【工具输出解析】{name}']
    try: data=json.loads(result) if isinstance(result,str) else result
    except (ValueError,TypeError): data=result
    if not isinstance(data,dict):
        lines.append(str(data))
        return '\n'.join(lines)
    if 'state' in data: lines.append('阶段：'+STATES.get(data['state'],str(data['state'])))
    for key,label in [('job_id','训练任务'),('export_id','导出任务'),('device','设备'),('gpu','GPU'),('message','说明'),('error','错误'),('output','输出文件'),('checkpoint','策略权重'),('output_dir','任务目录'),('container','容器'),('docker_state','容器状态'),('started_at','开始时间'),('elapsed_text','已用时间'),('training_text','训练用时'),('duration_text','总耗时'),('eta_text','预计剩余'),('summary_path','总结文件')]:
        if key in data: lines.append(f'{label}：{data[key]}')
    extraction=data.get('extraction')
    if isinstance(extraction,dict):
        stage={'smplx':'SMPL-X 拟合'}.get(extraction.get('stage'),'骨架提取')
        lines.append('%s：%s/%s（%s%%）'%(stage,extraction.get('current','?'),extraction.get('total','?'),extraction.get('percent','?')))
    if data.get('sent_report'):
        sent=data['sent_report']
        lines.append('主动报告：已发送；文件：%s'%(sent.get('path') or '未保存'))
    if data.get('summary'):
        lines.append('训练总结：\n'+str(data['summary']))
    jump=data.get('report')
    if isinstance(jump,dict) and 'success' in jump:
        lines.append(f"跳跃：成功={jump['success']}；摔倒={jump.get('fallen')}；连续腾空={jump.get('airborne_seconds')}秒；网络启用={jump.get('coordinator_used')}")
    if 'source_url' in data: lines.append('来源视频：'+str(data['source_url']))
    if data.get('skill'): lines.append('已登记技能：'+str(data['skill'].get('skill_id'))+'；物理回放通过不代表动作语义已核实。')
    if 'candidates' in data: lines.append('候选视频数量：'+str(len(data['candidates'])))
    if 'iteration' in data: lines.append(f"训练轮次：{data['iteration']}/{data.get('total_iterations','?')}")
    metrics=data.get('metrics',{})
    if isinstance(metrics,dict):
        for key,label in [('mean_reward','平均奖励'),('loss','损失'),('steps','累计采样步数')]:
            if key in metrics: lines.append(f'{label}：{metrics[key]}')
        if metrics.get('self_check'):
            check=metrics['self_check']; lines.append('权重自检：'+json.dumps({k:check.get(k) for k in ('before','after','changed','reason','model_error')},ensure_ascii=False))
        if metrics.get('curriculum'): lines.append('腿部课程：'+json.dumps(metrics['curriculum'],ensure_ascii=False))
    viewer=data.get('viewer')
    if isinstance(viewer,dict): lines.append(f"渲染窗口：{STATES.get(viewer.get('state'),viewer.get('state'))}；权重轮次：{viewer.get('checkpoint_iteration','—')}")
    evaluation=data.get('evaluation') or (metrics.get('evaluation') if isinstance(metrics,dict) else None)
    if not evaluation and 'fallen' in data: evaluation=data
    if isinstance(evaluation,dict):
        completed=evaluation.get('reference_completed',evaluation.get('completed'))
        lines.append(f"物理回放：摔倒={evaluation.get('fallen','未报告')}；参考完成={completed if completed is not None else '未报告'}")
        if evaluation.get('fallen'): lines.append('结论：物理回放失稳，训练未达标。')
        elif completed: lines.append('结论：参考片段已走完；仍需检查漂移和动作幅度，不能只凭完成判定学会原动作。')
        for key,label in [('simulation_seconds','仿真秒数'),('max_root_xy_error_m','最大平面误差(m)'),('joint_rmse_rad','关节RMSE(rad)'),('termination_reason','终止原因')]:
            if key in evaluation: lines.append(f'{label}：{evaluation[key]}')
        mse=evaluation.get('mean_joint_mse',evaluation.get('joint_mse'))
        if isinstance(mse,(float,int)) and math.isfinite(mse) and mse>=0: lines.append(f'关节RMSE：{math.sqrt(mse):.4f} rad')
    if 'leg_weight' in data:
        lines.append(f"腿部参考权重：{data['leg_weight']}（小于1表示缩幅，不是原视频全幅复现）")
    if data.get('state')=='completed' and not evaluation: lines.append('任务结束不等于动作训练达标。')
    lines.append('原始结构化输出：\n'+json.dumps(data,ensure_ascii=False,indent=2))
    return '\n'.join(lines)
