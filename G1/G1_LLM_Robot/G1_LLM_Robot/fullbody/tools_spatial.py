"""空间技能及节点技能联动工具schema。"""
def schemas():
    def tool(name,description,properties,required=()):
        return {'type':'function','function':{'name':name,'description':description,'parameters':{'type':'object','properties':properties,'required':list(required)}}}
    return [
        tool('robot_spatial_skill_list','列出按绝对世界关键点坐标保存的空间技能。',{}),
        tool('robot_spatial_skill_get','查询空间技能的关键点、时间轴、来源、关联节点技能和空间评估。',{'skill_id':{'type':'string'}},['skill_id']),
        tool('robot_spatial_skill_run','运行空间技能：spatial直接回放绝对关键点参考；node运行关联的原有节点权重策略；linked保留空间对照信息并运行空间参考。默认打开仿真窗口。',{'skill_id':{'type':'string'},'mode':{'type':'string','enum':['spatial','node','linked']},'viewer':{'type':'boolean'}},['skill_id']),
    ]
