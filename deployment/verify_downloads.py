"""Verify downloaded artifacts against publisher SHA256 metadata."""
import hashlib, json, os
from pathlib import Path
root=Path(os.environ.get('SAN_STORAGE','/media/pupu/PUPU/insPart/SAN'))
files={
 'weights/san_vit_large_14.pth':'dd605f75f8ec66a107164b064c3ef217deae2848349dc52e6eab6f0b071bf12c',
 'cache/clip/ViT-L-14-336px.pt':'3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02',
 'wheels/torch-2.1.0+cu118-cp310-cp310-linux_x86_64.whl':'a81b554184492005543ddc32e96469f9369d778dedd195d73bda9bed407d6589'}
results=[]
for name,expected in files.items():
 h=hashlib.sha256()
 with (root/name).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 actual=h.hexdigest()
 print(name,actual,'OK' if actual==expected else 'FAILED',flush=True)
 if actual!=expected:raise SystemExit('SHA256 mismatch: '+name)
 results.append(dict(path=name,sha256=actual,bytes=(root/name).stat().st_size))
Path(__file__).with_name('artifacts.json').write_text(json.dumps(results,indent=2)+'\n')
