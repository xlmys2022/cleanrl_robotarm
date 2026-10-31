"""共享的 MuJoCo 场景加载器（纯 MuJoCo, 不依赖 Gymnasium）。

为什么需要这个 loader:
    assets/scenes/panda_pick_place.xml 通过 <include> 复用只读的官方
    assets/robots/franka_panda/panda.xml。官方 panda.xml 自带 keyframe "home", 但它的
    qpos 只有 9 个分量（7 个臂关节 + 2 个 finger）。我们的场景加入了 cube 的 freejoint,
    nq 变成 16, MuJoCo 会以 "key 0: invalid qpos size, expected length 16" 拒绝编译。

    官方文件只读, 不能改; 所以这里在加载时把 panda.xml 文本里的 <keyframe> 块去掉,
    而 assets/scenes/panda_pick_place.xml 里已经定义了一份 16 维的 "home" 作为替代
    （前 9 维与官方完全一致, 后 7 维是 cube 的初始位姿）。

    注意: 由于这个 keyframe 冲突, 场景 XML 不能直接用 MjModel.from_xml_path 加载,
    必须经过本函数。
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCENE_XML = PROJECT_ROOT / "assets" / "scenes" / "panda_pick_place.xml"
PANDA_XML = PROJECT_ROOT / "assets" / "robots" / "franka_panda" / "panda.xml"
PANDA_MESH_DIR = PROJECT_ROOT / "assets" / "robots" / "franka_panda" / "assets"

# 场景文件里 include 的写法, 同时作为 from_xml_string 的 assets key
PANDA_INCLUDE = "../robots/franka_panda/panda.xml"

_KEYFRAME_BLOCK = re.compile(r"<keyframe>.*?</keyframe>", re.S)
_MESHDIR_ATTR = re.compile(r'meshdir="[^"]*"')


def load_pick_place_model(scene_xml: Path = SCENE_XML) -> mujoco.MjModel:
    """加载 Panda pick&place 场景, 返回 MjModel（官方 panda.xml 保持只读）。"""
    scene_text = Path(scene_xml).read_text(encoding="utf-8")
    if PANDA_INCLUDE not in scene_text:
        raise RuntimeError(f"场景文件里找不到 include: {PANDA_INCLUDE}")

    # 去掉官方 keyframe（见模块说明）; 其余内容原样复用
    panda_text = _KEYFRAME_BLOCK.sub("", PANDA_XML.read_text(encoding="utf-8"))

    # 从字符串编译时没有 modelfiledir, 相对 meshdir 会按当前工作目录解析,
    # 这里换成绝对路径, 保证在任何工作目录下都能找到 mesh。
    scene_text = _MESHDIR_ATTR.sub(f'meshdir="{PANDA_MESH_DIR.as_posix()}"', scene_text)

    return mujoco.MjModel.from_xml_string(scene_text, {PANDA_INCLUDE: panda_text.encode("utf-8")})
