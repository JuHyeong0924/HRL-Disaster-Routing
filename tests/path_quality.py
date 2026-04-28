#!/usr/bin/env python3
"""Ablation 경로 품질 분석 스크립트"""
import re, os

base = 'tests/ablation_results'
exps = ['BASELINE','A1','A2','A3','A4','A5','A6','A7',
        'S1','S2','S3','S4','S5','R1','R2','R3','R4','R5']

print(f"{'ID':>10} | {'EMA':>6} | {'AvgLen':>7} | {'AvgRw':>8} | {'AvgSucc':>7}")
print('-' * 60)
for d in exps:
    path = os.path.join(base, d, 'train_log.txt')
    if not os.path.exists(path):
        continue
    lines = open(path).readlines()
    # 마지막 500줄에서 PROGRESS_UPDATE 추출
    lens, rws, succs = [], [], []
    for l in lines[-500:]:
        m = re.search(r'Succ=([\d.]+)%.*Rw=(-?[\d.]+).*Len=([\d.]+)', l)
        if m:
            succs.append(float(m.group(1)))
            rws.append(float(m.group(2)))
            lens.append(float(m.group(3)))
    if lens:
        print(f'{d:>10} | {succs[-1]:5.1f}% | {sum(lens)/len(lens):7.1f} | {sum(rws)/len(rws):8.1f} | {sum(succs)/len(succs):5.1f}%')
