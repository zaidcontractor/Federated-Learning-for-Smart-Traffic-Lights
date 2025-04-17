# Federated-Learning-for-Smart-Traffic-Lights
This project aims to address urban traffic management challenges by leveraging Federated Learning combined with Reinforcement Learning to enable each region to train local AI models on real-time traffic data, while collaboratively contributing to a robust global model.

## 1 . Clone the repo & checkout the branch

```bash
git clone https://github.com/<org>/Federated-Learning-for-Smart-Traffic-Lights.git
cd Federated-Learning-for-Smart-Traffic-Lights
git checkout deep-DQN
```

---

## 2 . Create a Python 3.11 environment

```bash
# with conda (recommended; replace `mambaforge`/`miniconda` as needed)
conda create -n smarttl python=3.11 -y
conda activate smarttl
```

> **If you prefer `venv`/`pipenv`/Poetry**, that works too—just keep Python ≥ 3.10.

Install the Python requirements:

```bash
pip install torch tqdm gymnasium numpy pandas sumo-rl lxml
```

---

## 3 . Install SUMO **(choose ONE variant that matches your OS & patience)**  

| Option | macOS | Linux (Ubuntu/Debian) | Windows / WSL |
|--------|-------|-----------------------|---------------|
| **A. Pre‑built** (quickest) | `brew install sumo` | `sudo apt install sumo sumo-tools` | • Install **WSL 2 + Ubuntu** → use the Linux column. <br>• Or grab the official .msi from <https://sumo.dlr.de> |
| **B. Build from source** (10‑15 min, faster runtime, works on any OS) | see commands ↓ | see commands ↓ | build inside **WSL 2** (recommended) or MSYS2 / MinGW |

### B. Source‐build instructions (all OSes)

```bash
# 1) prerequisites (already present on most distros)
sudo apt update && sudo apt install -y git g++ cmake libxerces-c-dev libfox-1.6-dev \
    libproj-dev libgdal-dev libgl2ps-dev libopenscenegraph-dev libfftw3-dev \
    swig python3-dev python3-pip

# macOS equivalent (Homebrew):
brew install cmake xerces-c fox proj gdal openscenegraph fftw swig

# 2) build
git clone https://github.com/eclipse/sumo.git
cd sumo
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/sumo-local   # install to $HOME
make -j$(nproc)                                   # ~10 min
make install
```

### 3 . Add SUMO to your environment

*(replace the path if you used a different install location)*

```bash
# Bash / Z‑sh
echo 'export SUMO_HOME=$HOME/sumo-local/share/sumo' >> ~/.bashrc   # or ~/.zshrc
echo 'export PATH="$SUMO_HOME/bin:$PATH"'          >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
which netgenerate   # → …/bin/netgenerate
echo $SUMO_HOME     # → …/share/sumo
```

---

## 4 . (Only once) prepare the project folders

```bash
mkdir -p nets routes weights
```

No manual data download is required—the pre‑processed PeMS CSVs already live in  
`preprocessed_data/PeMS/`.

---

## 5 . Run the Deep DQN trainer

```bash
python deep_dqn.py \
  --csv preprocessed_data/PeMS \   # directory of many CSVs *or* a single file
  --env sumo_env \                 # don’t change unless you wrote another env
  --device cpu                     # or mps (Apple Silicon) / cuda (NVIDIA)
```

* The script will:
  * autogenerate a 2 × 2 grid network (`nets/4way.net.xml`);
  * patch in **exactly one** traffic light if SUMO forgot to add one;
  * pick a random PeMS CSV each episode, convert its first **10 minutes** into a
    tiny route file (keeps training snappy);
  * train a Dueling‑DQN agent for **900 simulation s** (~15 min) per episode;
  * save weights to `weights/local_tl.pt` when it finishes.

> **Tip:** want more/less data?  
> Edit **`sumo_env.py`** → function `_csv_to_route` → change  
> `df.iloc[:120]` (120 rows × 5 s = 10 min) to any slice you like.

---

## 6 . Typical training‑time vs. speed tips

| Bottleneck | Fix |
|------------|-----|
| Simulation crawls | • Use Apple‑silicon or CUDA device for the NN (`--device mps/cuda`).<br>• Source‑build SUMO (Option B) – ~2‑3 × faster on M‑series Macs. |
| Long episodes | Lower `num_seconds` in `sumo_env.py → _CsvEnv` (900 → 300). |
| Huge FLOWS | Trim CSV time range as shown above. |

---

## 7 . Troubleshooting checklist

| Symptom | Likely cause / solution |
|---------|-------------------------|
| `netgenerate: command not found` | `SUMO_HOME`/`PATH` mis‑set → re‑export (see §3). |
| `No TLS found`                   | Delete `nets/4way.net.xml` and re‑run (`sumo_env` patches it). |
| `Vehicle veh0 has no route`      | Old route files → `rm routes/*.rou.xml` and retry. |
| Python `KeyError (0,3)` in `traffic_signal.py` | Upgrade `sumo-rl` (`pip install -U sumo-rl`) – fixed in ≥ 1.4.5. |
