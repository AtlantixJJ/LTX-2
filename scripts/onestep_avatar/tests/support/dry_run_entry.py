"""Exact train and support dry-run probe bodies, isolated in child rows."""

SOURCE_0 = r"""
import sys
from pathlib import Path
from types import SimpleNamespace
from scripts.onestep_avatar import train
from scripts.onestep_avatar.training import engine
engine.prepare_run = lambda *_args, **_kwargs: (
    SimpleNamespace(membership={'sha256': 'a'*64}), {'sha256': 'b'*64, 'samples': []}, None, False)
engine.build_transformer = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('loaded transformer'))
output = Path(sys.argv[2])
assert train.main(['--mode', sys.argv[1], '--subset', '/boundary-only', '--output', str(output), '--dry-run']) == 0
assert not output.exists()
assert not any(name.startswith('scripts.onestep_avatar.experiments') for name in sys.modules)
"""

SOURCE_1 = r"""
import sys
from pathlib import Path
from scripts.onestep_avatar import evaluate, infer
owner, mode, destination = sys.argv[1:]
output = Path(destination)
if owner == 'evaluate':
    evaluate.prepare_evaluation = lambda _args: (
        None, [None], [(None, None, {'mode': mode}, None)], {})
    arguments = ['--mode', mode, '--subset', '/boundary-only', '--output', str(output),
                 '--variant', 'dev', '--guide-mode', 'd0', '--schedule', '0.725', '0', '--dry-run']
    status = evaluate.main(arguments)
else:
    infer.prepare_product = lambda _args: (None, None, None, None, {'mode': mode}, None)
    arguments = ['--mode', mode, '--guide', '/boundary-guide', '--first-image', '/boundary-image',
                 '--output', str(output), '--variant', 'dev', '--schedule', '0.725', '0', '--dry-run']
    status = infer.main(arguments)
assert status == 0
assert not output.exists()
assert not any(name.startswith('scripts.onestep_avatar.experiments') for name in sys.modules)
"""
