"""Read-only progress monitor; writes only the deployment progress log."""
import datetime
import os
from pathlib import Path
import time
root = Path('/home/pupu/insPart/SAN')
log = root / 'logs/progress.log'
storage = Path('/media/pupu/PUPU/insPart/SAN')
files = [('SAN-L', storage/'weights/san_vit_large_14.pth', 1821680110), ('CLIP', storage/'cache/clip/ViT-L-14-336px.pt', 934088680), ('PyTorch wheel', storage/'wheels/torch-2.1.0+cu118-cp310-cp310-linux_x86_64.whl', 2325888121)]
def emit(s):
    with log.open('a') as f:
        f.write(datetime.datetime.now().astimezone().isoformat(timespec='seconds')+' '+s+'\n')
emit('进度监控启动；下载文件达到预期大小后仍需校验，环境完成以实际验证结果为准。')
while True:
    for name,path,total in files:
        try: size=path.stat().st_size
        except OSError: size=0
        parts=path.parent/(path.name+'.parts')
        try: size+=sum(p.stat().st_size for p in parts.iterdir())
        except OSError: pass
        emit(f'{name} 下载: {size/1e6:.1f}/{total/1e6:.1f} MB ({100*size/total:.1f}%)')
    active=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            cmd=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
            args=cmd.split()
            if args and not args[0].endswith(('bash','sh')):
                if 'deployment/download_file.py ' in cmd: active.append('分段下载 '+args[-2].split('/')[-1])
                elif 'instructpart-san/bin/python -m pip' in cmd: active.append('安装 Python 依赖')
                elif cmd.startswith('curl '): active.append('下载文件')
        except OSError: pass
    emit('正在运行: '+' | '.join(active) if active else '没有检测到安装/下载进程；请结合阶段记录确认状态。')
    time.sleep(10)
