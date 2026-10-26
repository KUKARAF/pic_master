"""Run E3 variants through the harness (raw window cache is shared). usage: run_variants.py v1 v2 ... | deep"""
import json, os, sys
BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [BENCH, os.path.dirname(BENCH)]
import harness
from methods.e3_texture import E3Texture, VARIANTS
for v in sys.argv[1:]:
    if v == 'deep':
        from methods.e3_texture_deep import E3Deep
        m = E3Deep()
    elif v == 'fused':
        from methods.e3_texture_fused import E3Fused
        m = E3Fused()
    else:
        m = E3Texture(v)
    rep = harness.evaluate(m, harness.DATA_DIR, ['pattern'], False, None)
    harness.print_report(rep)
    json.dump(rep, open(os.path.join(BENCH, 'results', f'e3_{m.name.replace("e3_", "")}.json'), 'w'), indent=1)
