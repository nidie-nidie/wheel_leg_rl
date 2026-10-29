import sys

import torch

from debug.sim2sim import action_mode_engine_probe_20261020 as probe


# 先加载项目CUDA torch，再提供已安装的MuJoCo契约依赖。
assert torch.version.cuda is not None
sys.path.append(str(probe.SIM / '.venv/Lib/site-packages'))

if __name__ == '__main__':
    probe.main()
