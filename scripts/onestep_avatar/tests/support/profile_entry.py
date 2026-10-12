"""Exact isolated profile probe, run only in a child interpreter."""

SOURCE = r"""
import json,sys
from pathlib import Path
import torch
from scripts.onestep_avatar import LTX_ROOT, hashing
from scripts.onestep_avatar.execution import software
profile=sys.argv[1]
mode=None if sys.argv[4]=="none" else sys.argv[4]
input_path,output_path=map(Path,sys.argv[2:4])
entry_paths=[]
if profile=="training":
    from scripts.onestep_avatar.training import config,engine
    from scripts.onestep_avatar.corpus import dataset,subset
    from scripts.prune.core import model_registry
    settings=config.parse_settings(["--mode",mode,"--subset",str(input_path),"--output",str(output_path)])
    membership={"schema_version":2,"kind":subset.KIND,"objective":"bg","corpus_root":str(input_path.parent),
        "splits":{"train":["actor"]},"sources":[{"relative_dir":"actor/view","actor":"actor","split":"train",
        "n_latent_frames":7,"shape":[128,7,2,2],"fps":30}],"excluded":{}}
    membership["sha256"]=subset.membership_hash(membership)
    specification=model_registry.resolve("2.5")
    plan=config.build_frame_plan(settings,membership,specification.scale_factors)
    master=torch.arange(128*7*2*2,dtype=torch.float32).reshape(128,7,2,2)
    video=dataset.EncodedVideo("actor/view","actor","train",master,master.clone(),30,{}, {})
    grid,capture,guide,start,end=engine.tokens_for_sample(video,plan["samples"][0],settings,specification,
        torch.device("cpu"),step=1,rank=0,slot=0)
    assert capture.shape==guide.shape and end>start and grid.latent_frames==end-start
    entry_paths=["training.config.parse_settings","training.config.build_frame_plan","training.engine.tokens_for_sample"]
elif profile=="evaluation":
    from scripts.onestep_avatar import evaluate,metrics,previews
    args=evaluate.parse_args(["--mode",mode,"--subset",str(input_path),"--output",str(output_path),"--schedule","1","0"])
    from scripts.onestep_avatar.model import causal
    from ltx_core.types import SpatioTemporalScaleFactors
    geometry=causal.CausalGeometry(SpatioTemporalScaleFactors(time=8,height=32,width=32),block_latent_frames=2,context_latent_frames=8)
    assert geometry.block_latent_frames==2
    value=torch.zeros(1,128,3,1,1)
    assert metrics.encoded_metrics(value,value)["mse"]==0
    fixed={"input_files":{"noise":{"tensor_sha256":hashing.tensor_sha256(value)}}}
    previews.verify_preview_tensors(fixed,{"noise":value})
    try:
        previews.verify_preview_tensors(fixed,{"noise":value+1})
    except ValueError:
        pass
    else:
        raise AssertionError("changed fixed preview tensor was accepted")
    entry_paths=["evaluate.parse_args","metrics.encoded_metrics","model.causal.CausalGeometry","previews.verify_preview_tensors"]
elif profile=="inference":
    from scripts.onestep_avatar import infer
    args=infer.parse_args(["--mode",mode,"--guide",str(input_path),"--first-image",str(input_path),
        "--output",str(output_path),"--schedule","1","0"])
    guide=torch.zeros(1,128,7,2,2)
    record={"objective":"bg","fps":30,"box_xyxy":[0,0,64,64],"edge":64,
        "vae_fingerprint":"same-vae","encode_contract_version":2,"pixel_frames":1}
    infer.check_inputs(guide,guide[:,:,:1],record,{**record,"input_role":"supplied_image"})
    entry_paths=["infer.parse_args","infer.check_inputs"]
elif profile=="decoding":
    from scripts.onestep_avatar import comparisons,decode_saved,media
    decode_saved.parse_args(["--jobs",str(input_path),"--output",str(output_path)])
    media.native_decoder_settings()
    comparisons.parse_saved_comparison_args(["--render-saved-comparisons",str(input_path),"--output",str(output_path)])
    entry_paths=["decode_saved.parse_args","media.native_decoder_settings","comparisons.parse_saved_comparison_args"]
else:
    from scripts.onestep_avatar import prepare_inputs
    from PIL import Image
    image_path=input_path.with_suffix(".png")
    Image.new("RGB",(4,4),(19,37,53)).save(image_path)
    pixels=prepare_inputs.single_image(image_path)
    assert pixels.shape==(4,4,3) and pixels[0,0].tolist()==[19,37,53]
    if mode is None:
        prepare_inputs.parse_args(["supplied-image","--image",str(image_path),"--guide",str(input_path),
            "--output",str(output_path),"--gpu-id","0"])
        entry_paths=["prepare_inputs.parse_args: supplied-image","prepare_inputs.single_image"]
    else:
        from scripts.onestep_avatar import evaluate,previews
        arguments=["--mode",mode,"--subset",str(input_path),"--output",str(output_path),"--schedule","1","0"]
        args=evaluate.parse_args(arguments)
        prepared=prepare_inputs.preview_arguments(arguments,args,input_path,7)
        assert "--noise-file" in prepared
        prepare_inputs.parse_args(["preview","--references",str(input_path),"--output",str(output_path),
            "--gpu-id","0","--evaluation-arguments",*arguments])
        value=torch.zeros(1,128,3,1,1)
        previews.verify_preview_tensors({"input_files":{"noise":{"tensor_sha256":hashing.tensor_sha256(value)}}},
            {"noise":value})
        entry_paths=["prepare_inputs.parse_args: preview","prepare_inputs.preview_arguments",
            "previews.verify_preview_tensors"]
loaded=[]
for name,module in list(sys.modules.items()):
    if name.startswith("scripts.onestep_avatar") and getattr(module,"__file__",None) and ".tests" not in name:
        loaded.append(str(Path(module.__file__).resolve().relative_to(LTX_ROOT)))
print(json.dumps({"profile":profile,"mode":mode,"entry_paths":entry_paths,"loaded":sorted(loaded),
    "declared":software.source_files(profile,mode)}))
"""
