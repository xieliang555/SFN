from gym.envs.registration import register


register(
    id='peg-in-hole-v11',
    entry_point='gymEnv.envs.peg_in_hole_v11:PegInHole',
)

'''
peg-in-hole-v1: cylinder base 间隙分割
peg-in-hole-v2: cylinder base 角度匹配
peg-in-hole-v3: square base 间隙分割 (eye in hand)
peg-in-hole-v4: square base yaw角度匹配 (eye in hand)
peg-in-hole-v5: square base 间隙分割（eye to hand）
peg-in-hole-v6: square base yaw角度匹配 hole特征旋转(eye to hand)
peg-in-hole-v7: square base yaw角度匹配, 整个base作为mask,周围随机化(eye to hand)
peg-in-hole-v8: complex base yaw角度匹配, 渲染得到mask, 周围随机化(eye to hand)
peg-in-hole-v9: complex base yaw角度匹配, contrastive loss正则+强化学习(eye to hand)
peg-in-hole-v10: complex base xy位置匹配
peg-in-hole-v11: complex base xy位置匹配（角度由peg-in-hole-v8给定）
'''