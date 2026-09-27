import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
ROOT = REPO.parent / 'expr/onestep_avatar/d1_comparison/videos'
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for manifest in sorted(ROOT.rglob('manifest.json')):
    meta = json.loads(manifest.read_text())
    for video in meta['videos']:
        if abs(video['sigma'] - 0.909375) > 1e-7:
            continue
        path = manifest.parent / Path(video['output']).name
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            w = rgb.shape[1] // 3
            frames.append(np.stack([cv2.resize(rgb[:, i*w:(i+1)*w], (256,256), interpolation=cv2.INTER_AREA) for i in range(3)]))
        cap.release()
        arr = np.stack(frames).astype(np.float32) / 255
        boundaries = [(b[0]-1)*8+1 for b in video['blocks'][1:]]
        # Each transition t compares frames t-1 and t. A +/-2 window allows decoder spread.
        for arm in (0,1,2):
            values = []
            for t in range(1, len(arr)):
                mask = (arr[t,0].min(-1) < .93) | (arr[t-1,0].min(-1) < .93)
                mask = cv2.dilate(mask.astype(np.uint8), np.ones((7,7),np.uint8)).astype(bool)
                delta = arr[t,arm]-arr[t-1,arm]
                gt_delta = arr[t,0]-arr[t-1,0]
                values.append((t, float(np.abs(delta)[mask].mean()), float(np.abs(delta-gt_delta)[mask].mean())))
            for region in ('boundary_pm2','interior'):
                chosen = [v for v in values if v[0]>=5 and (min(abs(v[0]-b) for b in boundaries)<=2)==(region=='boundary_pm2')]
                rows.append(dict(video=str(path), trajectory=video['trajectory'], arm=['gt','d0','d1'][arm], region=region,
                    n=len(chosen), temporal_mae=float(np.mean([v[1] for v in chosen])),
                    temporal_residual_mae=float(np.mean([v[2] for v in chosen]))))
        if '0008_01_view00_cam51' in path.name and video['trajectory']=='official':
            ts=[14,16,17,19,30,32,33,35,46,48,49,51]
            sheet=Image.new('RGB',(len(ts)*160, 3*184),'white')
            draw=ImageDraw.Draw(sheet)
            for arm in range(3):
                for j,t in enumerate(ts):
                    sheet.paste(Image.fromarray(frames[t][arm]).resize((160,160)),(j*160,arm*184+24))
                    draw.text((j*160+4,arm*184+5),f'{["GT","D0","D1"][arm]} f{t}',fill='black')
            sheet.save(OUT/'official_sigma0909375_contact.png')
        print(path.name, len(frames), flush=True)
with (OUT/'boundary_metrics.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
summary=[]
for trajectory in ('official','one_step'):
    for arm in ('gt','d0','d1'):
        group={region:[r for r in rows if r['trajectory']==trajectory and r['arm']==arm and r['region']==region] for region in ('boundary_pm2','interior')}
        item={'trajectory':trajectory,'arm':arm,'views':len(group['interior'])}
        for metric in ('temporal_mae','temporal_residual_mae'):
            for region in group:
                item[metric+'_'+region]=float(np.mean([r[metric] for r in group[region]]))
        summary.append(item)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
