#!/usr/bin/env python3
"""Summarise retained f32 study logs: python3 f32_report.py <f32-logs dir>."""
import collections
import json
import math
import pathlib
import statistics
import sys

logs = pathlib.Path(sys.argv[1])
rows = collections.defaultdict(list)
for path in sorted(logs.glob('*.log')):
    if path.name not in {'accuracy.log', 'grid-speed.log', 'trace-anchor.log', 'engine.log'}:
        continue
    for line in path.open():
        if line.startswith('F32_'):
            tag, payload = line.split(' ', 1)
            rows[tag].append(json.loads(payload))

out = []
def write(s=''):
    out.append(s)
def table(headers, values):
    write('| ' + ' | '.join(headers) + ' |')
    write('|' + '|'.join('---' for _ in headers) + '|')
    for row in values:
        write('| ' + ' | '.join(str(x) for x in row) + ' |')
    write()
def fmt(x):
    return '—' if x is None else f'{x:.4g}'
def err(x, budget):
    return f'{x:.4g} ({x / budget:.3g}×)'
def pct(v, p):
    v = sorted(v)
    position = (len(v) - 1) * p
    i = math.floor(position)
    return v[i] + (v[min(i+1, len(v)-1)] - v[i]) * (position-i)
def dist(v):
    return '—' if not v else f'{statistics.median(v)*1000:.3f} / {pct(v,.95)*1000:.3f}'
def mode(r):
    return 'df' if r['precision']['opl_double_float'] else 'f32'

def timings(tag, keys, modes):
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows[tag]:
        if r['phase'] == 'measured':
            groups[tuple(r[k] for k in keys)][r['mode']].append(r)
    values=[]
    for key, data in sorted(groups.items()):
        cells=[]
        for m in modes:
            samples=data[m]
            cells += [dist([r['wall_seconds'] for r in samples]), dist([r['device_seconds'] for r in samples]) if any(r['device_seconds'] for r in samples) else '—']
        values.append([*key, *cells, '/'.join(str(len(data[m])) for m in modes)])
    table([*keys, *[label for m in modes for label in (f'{m} wall',f'{m} device')], 'n by mode'],values)
    return groups

write('# Pure f32 Metal: measured cost and accuracy')
write()
write('2026-11-02; Apple M2 Pro. All timings are provisional under normal host load. Units: mm, direction cosines, waves, and milliseconds for timing tables. DF = strict double-float; f32 = strict all-single-precision arithmetic. Parenthesised × values are fractions of the whole GPU §9.4 budget, not the one-third sampled qualification margin.')
write()
write('## Implementation and reproduction')
write()
write('`MKLIGHT_PURE_F32=1` drops hi/lo residuals, error-free transforms and compensation without changing the packed ABI, trace/refinement algorithms, reduction trees or strict compilation contract. Both offline metallibs are embedded; runtime source has the same define. Default `MetalOptions` and `MetalBackend::new()` remain DoubleFloat, including engine Auto. Explicit callers use `MetalOptions::from_env()` to opt into `MKLIGHT_METAL_PRECISION=f32`; `df` selects DoubleFloat. Kernel-build metadata and precision descriptor booleans distinguish the variants. CPU-f32 emulation remains the DF specification; its bitwise tests use DF.')
write()
write('```sh\nsource ~/.cargo/env\nexport RUSTC_WRAPPER=\ncargo test --release -p mklight-metal --test f32_study trace_and_aiming_accuracy -- --ignored --nocapture\ncargo test --release -p mklight-metal --test f32_study trace_grid_and_aiming_speed -- --ignored --nocapture\nMKLIGHT_METAL_PRECISION=f32 cargo test --release -p mklight-metal --test trace -- --ignored --nocapture\ncargo test --release -p mklight-engine --test f32_study -- --ignored --nocapture\npython3 crates/mklight-metal/tests/f32_report.py <f32-logs-directory>\n```')
write()
write('Real Metal execution requires access to the macOS GPU; a sandbox run with no device is not evidence. Offline compilation keeps its module cache inside Cargo OUT_DIR. `RUSTC_WRAPPER=` avoids a compiler-cache daemon outside the workspace sandbox.')
write()
write('## Geometry, OPL and OPD: all 16 designs')
write()
write('The §5.1 64×64 pupil-grid harness covers all fields and wavelengths, all-surface positions/directions and image OPD. It counts every status outcome, including non-survivors; numerical maxima describe comparable rays. OPL is separately checked on all live per-surface records at the reported point (CONVENTIONS §12). OPL fractions use the tighter internal 1e-7 mm target; OPD uses 5e-3 waves. Raw reports retain compared/ray counts, bands and failure diagnostics.')
write()
trace=collections.defaultdict(dict)
for r in rows['F32_TRACE']:
    trace[r['sample']][mode(r)] = r['report']
values=[]
for sample,data in trace.items():
    values.append([sample,*[err(data[m][q],budget) for q,budget in [('max_position_mm',2e-5),('max_direction',2e-6),('max_opd_waves',5e-3)] for m in ('df','f32')],*[f"{data[m]['survival_mismatches']}/{data[m]['band_mismatches']}/{data[m]['chief_mismatches']}" for m in ('df','f32')]])
table(['design','DF pos','f32 pos','DF dir','f32 dir','DF OPD','f32 OPD','DF status/band/chief','f32 status/band/chief'],values)
opl=collections.defaultdict(dict)
for r in rows['F32_OPL']:
    opl[r['sample']][r['mode']]=r
values=[]
for sample,d in opl.items():
    values.append([sample,*[err(d[m]['max_mm'],1e-7) for m in ('df','f32')],*[d[m]['compared'] for m in ('df','f32')]])
table(['design','DF OPL mm (target fraction)','f32 OPL mm (target fraction)','DF records','f32 records'],values)
write('## Real aiming')
write()
write('Same grid of 7×7 pupil points, every field/wavelength, real aiming at requested relative tolerance 1e-10. Independently retrace generated rays to the stop in CPU f64, including rays marked failed by device aiming; failure counts are not discarded from the study. A zero device residual can conceal an f32 cancellation error, so independently measured residuals are the accuracy result. Timing uses generation/aiming plus full trace; CPU f64 uses the same rays. Device failures remain counted and are not successful solutions.')
write()
aim=collections.defaultdict(dict)
for r in rows['F32_AIM']:
    aim[r['sample']][r['mode']]=r
values=[]
for sample,d in aim.items():
    values.append([sample,*[err(d[m]['max_relative_stop_residual'],1e-10) for m in ('df','f32')],*[f"{d[m]['failed']}/{d[m]['compared']}/{d[m]['stop_failed']}" for m in ('df','f32')]])
table(['design','DF residual / requested','f32 residual / requested','DF failed/compared/stop-failed','f32 failed/compared/stop-failed'],values)
write('## Trace anchor: 10,000 and 1,000,000 rays')
write()
write('Double Gauss, 12 surfaces, exactly stated ray counts; three fields, primary wavelength. One warmup, ten alternating measured rounds (order reverses each round); median / interpolated p95. Wall includes input-plane cloning, upload, dispatch, output assembly and destruction. Device is completed Metal command-buffer timestamp time. CPU uses independent f64 tracing with Rayon; input generation and immutable-system upload are outside timing on both GPU variants.')
write()
timings('F32_TRACE_TIME',['rays'],['df','f32','cpu_f64'])
write('## Trace-grid and aiming speed: matched accuracy workloads')
write()
write('One warmup plus ten alternating measured rounds. Modes 0/1/2 mean DF/f32/CPU f64. Trace-grid includes all-surface record output; aiming uses the same all-field/wavelength 7×7 points as its error study. Aiming plan compilation and immutable upload are outside warm timing, while output readback is inside. Timing is not an accuracy pass.')
write()
timings('F32_GRID_TIME',['sample'],[0,1,2])
timings('F32_AIM_TIME',['sample'],[0,1,2])
write('## Engine analyses: matched sampling and unchanged qualification')
write()
write('All 16 designs and all non-optimisation GPU registry analyses. Geometric analyses use defaults. Wave pupils use grid 64 with the default 65-ray lattice; PSF/EE/LSF use image size 256. PSF spacing is frozen from independent CPU first-order working F/# and shortest wavelength / 2.5; image simulation uses PSF size 128 and 0.001 mm spacing. MTF uses max frequency 50 cycles/mm; field/focus MTF uses [10,20,40] cycles/mm with focus range ±0.1 mm. All other settings are retained in each RunInfo. Fixed sampling deliberately measures precision separately from automatic-grid changes; sampling/truncation warnings remain in the logs and these are differential accuracy results, not new wave-sampling qualification claims. LSF/EE retain the algorithms’ working-NA spacing; coordinate errors are reported separately.')
write()
write('Each combination first runs the normal explicit Metal session with unchanged geometry and complete-chain gates, retaining its verdict. A separate ignored test invokes the same public native adapters directly to observe rejected f32 results; this bypass exists only in the study, not production. Raw failures, undefined/mask/array-shape differences and qualification fallbacks are reported explicitly. A normal-session CPU fallback is never counted as f32 accuracy or device speed. Both raw and session timings use one warmup and ten alternating rounds. Raw adapter wall includes native setup and final output packing; session wall additionally includes selection and cached qualification/fallback.')
write()
gates=collections.defaultdict(dict)
for r in rows['F32_ENGINE_GATE']:
    gates[(r['sample'],r['analysis'])][r['mode']]=r
values=[]
for (sample,analysis),d in sorted(gates.items()):
    values.append([sample,analysis,*[d.get(m,{}).get('used','ERROR') for m in (0,1)],*[d.get(m,{}).get('reason',d.get(m,{}).get('error','')) or '' for m in (0,1)]])
table(['design','analysis','DF normal-session used','f32 normal-session used','DF reason','f32 reason'],values)
write('### Raw native-adapter times (ms median / p95)')
write()
timings('F32_ENGINE_TIME',['sample','analysis'],[1,2,0])
write('### Complete session times, with reported fallbacks (ms median / p95)')
write()
timings('F32_ENGINE_SESSION_TIME',['sample','analysis'],[1,2,0])
write('### Per-quantity end-result error maxima over designs')
write()
write('Each cell gives maximum absolute error and the largest error/budget fraction (which may occur at a different design/reference value for relative budgets). Discrete/undefined/shape mismatches are counted separately and never treated as zero numerical error. A dash budget means §9.4 does not define a direct scalar budget for that metadata quantity. Image/PSF L1 and energy metrics are normalised to reference L1/energy; the LSF integral uses the same matched-grid sum. Stray class flux, map pixels and moments use five combined standard errors (no extra quadrature allowance); the energy ledger uses 1e-5 of emitted power. Radial quantiles use the smallest positive reference radial interval. All numeric result fields and buffers are examined. Detailed per-design records are in engine.log; whole budgets are used here. This report does not reinterpret unknown metadata as optical positions.')
write()
quantities=collections.defaultdict(lambda:collections.defaultdict(lambda:{'max_error':0.,'budget_fraction':None,'mismatches':0,'compared':0,'designs':0}))
for r in rows['F32_ENGINE_ERROR_STUDY']:
    for quantity,value in r['quantities'].items():
        q=quantities[(r['analysis'],quantity)][r['mode']]
        q['max_error']=max(q['max_error'],value['max_error'])
        if value['budget_fraction'] is not None:
            q['budget_fraction']=max(q['budget_fraction'] or 0,value['budget_fraction'])
        q['mismatches']+=value['mismatches'];q['compared']+=value['compared'];q['designs']+=1
values=[]
for (analysis,quantity),data in sorted(quantities.items()):
    values.append([analysis,quantity,*[f"{fmt(data[m]['max_error'])} ({fmt(data[m]['budget_fraction'])}×)" for m in (1,2)],*[data[m]['mismatches'] for m in (1,2)],*[data[m]['designs'] for m in (1,2)]])
table(['analysis','quantity','DF max (budget)','f32 max (budget)','DF mismatches','f32 mismatches','DF designs','f32 designs'],values)
write('### Raw adapter failures')
write()
errors=collections.defaultdict(lambda:[0,set()])
for r in rows['F32_ENGINE_ERROR']:
    key=(r['sample'],r['analysis'],r['mode']);errors[key][0]+=1;errors[key][1].add(r['error'])
table(['design','analysis','mode','failed attempts','reason'],[[*k,v[0],'; '.join(sorted(v[1]))] for k,v in sorted(errors.items())])
write('## Validation, artifacts and limits')
write()
write('Validation commands and complete results are retained in [fmt.log](f32-logs/fmt.log), [clippy.log](f32-logs/clippy.log), [workspace.log](f32-logs/workspace.log), [release.log](f32-logs/release.log). Both embedded metallibs compiled with `-Wall -Werror -fno-fast-math -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise -ffp-contract=off`. The ordinary test suite exercises the unchanged DF emulation spec; the f32 test checks independently known cancellation, ignored residual inputs, zero OPL low planes, deterministic repeated traces and both library origins.')
write()
write('Timing observations describe one M2 Pro and normal host load, not a quiet-host release policy. This experiment changes neither Auto cost rules nor qualification budgets. It does not qualify arbitrary designs, optimisation candidates or all possible analysis settings. The 16-design grid is a finite regression sample; the raw adapter deliberately exposes out-of-budget and discrete errors. Default auto-sampling PSF pilot evidence is retained separately, because changed grids do not support a pixelwise L1 comparison.')
write()
write('### Raw logs and identities')
write()
for name in ['host.json','accuracy.log','grid-speed.log','trace-anchor.log','engine.log','engine-pilot.log','fmt.log','clippy.log','workspace.log','release.log','study-commands.sh','study-commit.txt']:
    if (logs/name).exists():
        write(f'- [{name}](f32-logs/{name}) ({(logs/name).stat().st_size:,} bytes)')
write()
write('### Raw trace accuracy rows')
write()
write('```jsonl')
for tag in ['F32_ENV','F32_TRACE','F32_OPL','F32_AIM']:
    for r in rows[tag]:
        write(tag+' '+json.dumps(r,separators=(',',':')))
write('```')
(logs.parent/'f32-evidence.md').write_text('\n'.join(out)+'\n')
(logs/'summary.json').write_text(json.dumps({'trace':trace,'opl':opl,'aim':aim,'quantities':[{'analysis':k[0],'quantity':k[1],'modes':v} for k,v in sorted(quantities.items())],'counts':{k:len(v) for k,v in rows.items()}},indent=2)+'\n')
print(f'Wrote {logs.parent / "f32-evidence.md"}; counts: '+str({k:len(v) for k,v in rows.items()}))
