"""可直接加入 DeepSeek Function Calling 的视频工具。"""

def schemas():
    return [
        {'type':'function','function':{
            'name':'robot_learn_video',
            'description':'读取用户指定的本地单人视频，提取并保存动作，可立即由G1模仿上半身动作。腿部使用已有平衡策略；这不是全身模仿学习训练。遮挡帧保持上一姿态。返回真实回放状态、是否摔倒和漂移。',
            'parameters':{'type':'object','properties':{
                'video':{'type':'string','description':'用户提供的本地视频路径'},
                'output':{'type':'string','description':'动作npz输出路径，默认motions/video_pose.npz'},
                'max_seconds':{'type':'number','minimum':.1,'maximum':60,'description':'最多读取的秒数，默认30'},
                'play':{'type':'boolean','description':'是否立即回放，默认true'}},'required':['video']}}},
        {'type':'function','function':{
            'name':'robot_play_motion','description':'回放已提取的本地npz视频动作文件，返回仿真实测结果。',
            'parameters':{'type':'object','properties':{
                'motion':{'type':'string'},'speed':{'type':'number','minimum':.25,'maximum':2}},'required':['motion']}}}
    ]
