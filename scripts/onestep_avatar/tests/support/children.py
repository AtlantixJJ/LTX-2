"""Batch CLI rows and fork clean package imports from a warm third-party child.

Each row captures its own status/stdout/stderr. Forks inherit third-party imports
only, never scripts.*; package-import inventories therefore remain isolated.
"""

import json
import os
import subprocess
import sys

from scripts.onestep_avatar import LTX_ROOT, PACKAGE_ROOT


def normal_environment() -> object:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("ONESTEP_AVATAR_BLOCK_EXPERIMENTS", None)
    return env


def blocked_environment() -> object:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", ONESTEP_AVATAR_BLOCK_EXPERIMENTS="1")
    hook = str(PACKAGE_ROOT / "tests/blocked_imports")
    paths = env.get("PYTHONPATH", "").split(os.pathsep)
    env["PYTHONPATH"] = os.pathsep.join([hook, *(path for path in paths if path and path != hook)])
    return env


def run_dry_run_rows(cases: object, env: object) -> object:
    source = """
import contextlib,io,json,sys,traceback
cases=json.loads(sys.argv[1])
for case in cases:
 out,err=io.StringIO(),io.StringIO()
 code=0
 try:
  with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
   sys.argv=case['argv']
   exec(case['source'], {'__name__':'__main__'})
 except SystemExit as error:
  code=error.code or 0
 except BaseException:
  code=1
  traceback.print_exc(file=err)
 print(json.dumps({'id':case['id'],'returncode':code,'stdout':out.getvalue(),'stderr':err.getvalue()}))
"""
    run = subprocess.run(
        [sys.executable, "-c", source, json.dumps(cases)],
        cwd=LTX_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return [json.loads(line) for line in run.stdout.splitlines()]


def run_help_rows(modules: object, env: object) -> object:
    cases = [
        {
            "id": module,
            "argv": [module, "--help"] + (["--mode", "bidirectional"] if module.endswith(".train") else []),
            "source": "import runpy\nrunpy.run_module(" + repr(module) + ",run_name='__main__')",
        }
        for module in modules
    ]
    return run_dry_run_rows(cases, env)


def fork_profiles(profiles: object, env: object = None) -> object:
    source = """
import json,os,sys,tempfile,traceback
import torch,peft,transformers,ltx_core,safetensors,PIL
assert not any(name.startswith('scripts.') for name in sys.modules)
for case in json.loads(sys.argv[1]):
 with tempfile.TemporaryFile(mode='w+') as output:
  pid=os.fork()
  if pid==0:
   try:
    sys.stdout=output
    sys.argv=case['argv']
    exec(case['source'], {'__name__':'__main__'})
    output.flush()
    os._exit(0)
   except BaseException:
    traceback.print_exc(file=output)
    output.flush()
    os._exit(1)
  _,status=os.waitpid(pid,0)
  output.seek(0)
  print(json.dumps({'id':case['id'],'returncode':os.waitstatus_to_exitcode(status),'stdout':output.read()}))
"""
    run = subprocess.run(
        [sys.executable, "-c", source, json.dumps(profiles)],
        cwd=LTX_ROOT,
        env=env or normal_environment(),
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return [json.loads(line) for line in run.stdout.splitlines()]
