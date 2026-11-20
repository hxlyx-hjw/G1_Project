"""运行时资源定位与模型改造。

负责两件事
----------
1. **定位资源**：找到 menagerie 的 ``unitree_g1`` 目录和 RL 策略 ``motion.pt``。
   支持三种来源：显式参数 > 环境变量 > 用户主目录下的常见位置。
2. **改造模型**：把 menagerie 的 29 自由度 MJCF 改写成
   「腿部 12 个 <motor> + 腰/手臂 17 个 <position>」的混合模型，
   写到临时目录后再交给 MuJoCo 加载（MuJoCo 只能从文件加载，不能从字符串）。

为什么要改造成混合模型（关键设计）
----------------------------------
* RL 策略只关心 12 个腿关节的角度/角速度 + 躯干姿态，**不关心手臂**。
* 所以让腿执行器变成力矩电机（吃策略输出），手臂保留模型自带的位置伺服，
  就能在同一个模型上"边走边摆手臂"。
* 改造只动 ``<actuator>`` 段：运动学树、质量、惯量、网格全部保持不变，
  因此策略可以直接迁移（实测：12dof 前进 4.57m/10s → 29dof 前进 3.79m/10s）。
"""

from __future__ import annotations

import os
import tempfile

from . import config as C


# ======================================================================
# 异常
# ======================================================================
class ResourceNotFound(FileNotFoundError):
    """找不到模型或策略权重时抛出，附带可操作的提示。"""


# ======================================================================
# 一、资源定位
# ======================================================================
def _first_existing(paths):
    """返回第一个"存在且是文件"的路径；都没有则返回 None。"""
    for p in paths:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return None


def find_menagerie(explicit: str | None = None) -> str:
    """定位 ``mujoco_menagerie/unitree_g1`` 目录，返回绝对路径。

    查找顺序：
        1. 显式传入的 ``explicit``
        2. 环境变量 ``MUJOCO_MENAGERIE`` 指向的 menagerie 根
        3. 用户主目录下的若干常见克隆位置
    """
    home = os.path.expanduser("~")
    cands: list[str] = []

    if explicit:
        cands.append(explicit)

    env_root = os.environ.get(C.ENV_MENAGERIE)
    if env_root:
        cands.append(os.path.join(env_root, C.MENAGERIE_SUBDIR))
        cands.append(env_root)          # 也允许直接指向 unitree_g1 本身

    for sub in C.MENAGERIE_CANDIDATES:
        cands.append(os.path.join(home, *sub.split("/"), C.MENAGERIE_SUBDIR))

    for c in cands:
        if c and os.path.isfile(os.path.join(c, "g1.xml")):
            return os.path.abspath(c)

    raise ResourceNotFound(
        "找不到 mujoco_menagerie/unitree_g1（需要其中的 g1.xml 与 assets/）。\n"
        "解决办法任选其一：\n"
        "  * 克隆仓库: git clone https://github.com/google-deepmind/mujoco_menagerie\n"
        "  * 用参数指定: --menagerie <路径>\n"
        "  * 设环境变量: set MUJOCO_MENAGERIE=<menagerie 根目录>\n"
        "已尝试的位置:\n  " + "\n  ".join(cands))


def find_policy(explicit: str | None = None) -> str:
    """定位 RL 策略权重 ``motion.pt``，返回绝对路径。

    查找顺序同 :func:`find_menagerie`。
    """
    home = os.path.expanduser("~")
    cands: list[str] = []

    if explicit:
        cands.append(explicit)

    env = os.environ.get(C.ENV_POLICY)
    if env:
        cands.append(env)

    for sub in C.POLICY_CANDIDATES:
        cands.append(os.path.join(home, *sub.split("/")))

    got = _first_existing(cands)
    if got:
        return got

    raise ResourceNotFound(
        "找不到 RL 策略权重 motion.pt。\n"
        "它来自 unitree_rl_gym：\n"
        "  <unitree_rl_gym>/deploy/pre_train/g1/motion.pt\n"
        "解决办法任选其一：\n"
        "  * 用参数指定: --policy <motion.pt 路径>\n"
        "  * 设环境变量: set G1_POLICY_PATH=<motion.pt 路径>\n"
        "已尝试的位置:\n  " + "\n  ".join(cands))


# ======================================================================
# 二、模型改造
# ======================================================================
#: 场景文件模板。%(model)s 会被替换成机器人模型文件名。
_SCENE_TEMPLATE = """<mujoco model="g1 walkarm scene">
  <include file="%(model)s"/>

  <statistic center="0 0 0.6" extent="1.2"/>

  <visual>
    <headlight diffuse="0.65 0.65 0.65" ambient="0.28 0.28 0.28" specular="0.9 0.9 0.9"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="135" elevation="-24" offwidth="1920" offheight="1080"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""

#: 生成后的文件名
MODEL_FILENAME = "g1_29dof_walkarm.xml"
SCENE_FILENAME = "scene.xml"


def _patch_model_xml(src: str, assets_dir: str) -> tuple[str, int]:
    """把模型 XML 文本改造成混合执行器版本。

    参数
        src        menagerie ``g1.xml`` 的原始文本
        assets_dir 网格目录的绝对路径（会写进 ``meshdir``）

    返回
        (改造后的文本, 被替换的执行器个数)
    """
    n = 0
    for joint in C.LEG_ACTUATOR_NAMES:
        old = '<position class="g1" name="%s_joint" joint="%s_joint"/>' % (joint, joint)
        new = '<motor name="%s_joint" joint="%s_joint"/>' % (joint, joint)
        if old in src:
            src = src.replace(old, new)
            n += 1

    # meshdir 改成绝对路径：生成的文件在临时目录，相对路径会找不到网格
    src = src.replace('<compiler angle="radian" meshdir="assets"/>',
                      '<compiler angle="radian" meshdir="%s"/>'
                      % assets_dir.replace("\\", "/"))

    # 顺手改一下模型名，便于在报错信息里区分
    src = src.replace('<mujoco model="g1_29dof_rev_1_0">',
                      '<mujoco model="g1_29dof_walkarm">')
    return src, n


def build_walkarm_model(menagerie_dir: str, out_dir: str | None = None) -> str:
    """生成"腿力矩 + 臂位置伺服"的模型与场景，返回**场景文件**的绝对路径。

    参数
        menagerie_dir  ``unitree_g1`` 目录
        out_dir        输出目录；默认写到系统临时目录下
    """
    from pathlib import Path
    src = Path(menagerie_dir, "g1.xml").read_text(encoding="utf-8")
    assets = os.path.join(menagerie_dir, "assets")

    patched, n = _patch_model_xml(src, assets)
    if n != len(C.LEG_ACTUATOR_NAMES):
        raise RuntimeError(
            "腿部执行器改造失败：预期替换 %d 个，实际 %d 个。"
            "menagerie 版本可能已更新，actuator 写法变了。"
            % (len(C.LEG_ACTUATOR_NAMES), n))

    out_dir = out_dir or os.path.join(tempfile.gettempdir(), C.TMP_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, MODEL_FILENAME)
    scene_path = os.path.join(out_dir, SCENE_FILENAME)

    with open(model_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(patched)
    with open(scene_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(_SCENE_TEMPLATE % dict(model=MODEL_FILENAME))

    return os.path.abspath(scene_path)
