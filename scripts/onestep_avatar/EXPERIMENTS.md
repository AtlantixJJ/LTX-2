```bash
python -m scripts.onestep_avatar.execution.queue \
  --jobs /home/jianjinx/data2/VideoDiffusionModels/expr/onestep_avatar/runs/d0_bidirectional_sigma0.9/jobs.json \
  --state /home/jianjinx/data2/VideoDiffusionModels/expr/onestep_avatar/runs/d0_bidirectional_sigma0.9/state.json \
  --foreground --once --process-ledger /home/jianjinx/data2/VideoDiffusionModels/expr/onestep_avatar/processes.json \
  --gpu-pool 0,1,2,3
```


```bash
cd /data1/users/jianjinx/VideoDiffusionModels/LTX-2

PY=/home/jianjinx/data2/miniconda3/envs/ltx/bin/python
D=/home/jianjinx/data2/VideoDiffusionModels/expr/onestep_avatar/runs/d0_bidirectional_sigma0.9

CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMP_NUM_THREADS=8 \
MKL_NUM_THREADS=8 \
"$PY" -m accelerate.commands.launch \
  --config_file scripts/onestep_avatar/configs/fsdp.yaml \
  --num_processes 4 \
  --main_process_port 29785 \
  -m scripts.onestep_avatar.train \
  --mode bidirectional \
  --subset "$D/membership_all.json" \
  --output "$D/run_csv" \
  --model 2.5 \
  --variant dev \
  --guide-mode d0 \
  --objective white \
  --sigma-levels \
    0.897735 0.884479 0.869858 0.853651 0.835584 0.815319 0.792427 0.766363 0.736418 0.701656 0.660813 0.612141 0.553148 0.480164 0.387541 \
  --span-latent-frames 7 \
  --start-policy clip_start \
  --lora-rank 32 \
  --lora-alpha 32 \
  --lora-target attn \
  --lr 1e-4 \
  --warmup-steps 20 \
  --max-grad-norm 1 \
  --steps 1000 \
  --save-initial \
  --save-every 200 \
  --seed 42 \
  --noise-policy fresh \
  2>&1 | tee "$D/train.log"
```