1. How to download model.
```
python3 -m venv /scratch/atul_prakash/envs/hf
source /scratch/atul_prakash/envs/hf/bin/activate
pip install -U huggingface_hub
hf download zy12123/Food-R1 --local-dir /scratch/atul_prakash/models/Food-R1
```

2. How to check the model...
```
python3 -m venv /scratch/atul_prakash/models/foodr1
source /scratch/atul_prakash/models/foodr1/bin/activate
pip install torch torchvision
pip install "transformers==4.57.6" accelerate qwen-vl-utils
```

```
nvidia-smi
# get GPU free gpu x
export CUDA_VISIBLE_DEVICES=x
```

```
python3 - <<'PY'
import torch
from transformers import Qwen3VLForConditionalGeneration

model_path = "/scratch/atul_prakash/models/Food-R1"

print("Loading Food-R1...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("\nFood-R1 loaded successfully!")
print("GPU:", torch.cuda.get_device_name(0))
print("Allocated:", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")
print("Reserved:", round(torch.cuda.memory_reserved() / 1024**3, 2), "GB")
PY
```

3. how to install vllm to serve model.
```
python3 -m venv /scratch/atul_prakash/envs/vllm
source /scratch/atul_prakash/envs/vllm/bin/activate
pip install -U vllm --extra-index-url https://download.pytorch.org/whl/cu130
```

```
nvidia-smi
# get GPU free gpu x
export CUDA_VISIBLE_DEVICES=x
```

```
tmux new -s server
source /scratch/atul_prakash/envs/vllm/bin/activate
```

```
vllm serve /scratch/atul_prakash/models/Food-R1 \
  --served-model-name Food-R1 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

```
curl http://localhost:8000/v1/models
```

```
ssh -L 8888:localhost:8000 atul_prakash@10.1.7.58
```

```
http://localhost:8000/v1/models
```

4. How to install `Open WebUI`
```
python3 -m venv /scratch/atul_prakash/envs/web
source /scratch/atul_prakash/envs/web/bin/activate
pip install open-webui
```

```
open-webui serve --host 0.0.0.0 --port 3000
```

```
ssh -L 3000:localhost:3000 atul_prakash@10.1.7.58
```

```
http://localhost:3000/
```

```
Settings -> Connections -> OpenAI API -> Add
URL = http://localhost:8001/v1
Save
```

5. Install CloudFare Tunnel.
```
mkdir /scratch/atul_prakash/bin
cd /scratch/atul_prakash/bin
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg-deb -x cloudflared-linux-amd64.deb /tmp/cloudflared
cp /tmp/cloudflared/usr/bin/cloudflared /scratch/atul_prakash/bin/
echo 'export PATH="/scratch/atul_prakash/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
cloudflared --version
```

```
cloudflared tunnel login
```

```
cloudflared tunnel create food
```

```
cat > ~/.cloudflared/config.yml <<'EOF'
tunnel: 56e67c20-5664-4648-a852-142aa199d0d0
credentials-file: /home/atul_prakash/.cloudflared/56e67c20-5664-4648-a852-142aa199d0d0.json

ingress:
  - hostname: food.wily.in
    service: http://127.0.0.1:3000

  - service: http_status:404
EOF
```

```
cloudflared tunnel route dns food food.wily.in
```

```
cloudflared tunnel run food
```

# To Run

```
tmux new -s server
tmux new -s client
tmux new -s cf
CRTL+D
tmux ls
```

```
tmux a -t server
```

```
source /scratch/atul_prakash/models/vllm/bin/activate
```

```
nvidia-smi
```

```
export CUDA_VISIBLE_DEVICES=
```

```
vllm serve /scratch/atul_prakash/models/Food-R1 \
  --served-model-name Food-R1 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

```
tmux a -t client
```

```
source /scratch/atul_prakash/models/web/bin/activate
open-webui serve --host 0.0.0.0 --port 3000
```

```
tmux a -t client
```

```
cloudflared tunnel run food
```