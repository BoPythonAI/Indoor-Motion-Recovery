# Setup and reproduction

## 1. View or verify the release

The gallery and artifact verification require no simulator:

```bash
git clone https://github.com/BoPythonAI/Indoor-Motion-Recovery.git
cd Indoor-Motion-Recovery
python tools/verify_release.py
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000`.

## 2. Download the selected checkpoint

With the GitHub CLI:

```bash
mkdir -p models
gh release download v1.0.0 \
  --repo BoPythonAI/Indoor-Motion-Recovery \
  --pattern h1-bc-ema.pt --dir models
```

Expected SHA256:

```text
b3098abe5d13fda49c29d785f566f9fa5c96e332231ec105b8e33bc70a0ac4e2
```

The checkpoint is the unmodified selected training artifact. The runtime uses EMA parameters and validates the feature, action, reference and geometry contracts. The frozen flat and terrain priors are separate dependencies.

## 3. Rebuild the English videos

```bash
python -m pip install -r requirements-media.txt
gh release download v1.0.0 \
  --repo BoPythonAI/Indoor-Motion-Recovery \
  --pattern demo_sources.zip --dir downloads
python -m zipfile -e downloads/demo_sources.zip downloads
python tools/render_demos.py \
  --inputs downloads/demo_sources/inputs.json \
  --output rebuilt_media
```

The source archive contains nine unannotated rendered clips plus audited image-space anchors, force/contact values and source identities. It contains rendered media, not raw scene meshes or a complete simulation dataset.

The renderer preserves all source frames, adds English annotations and produces three comparison MP4s, nine individual MP4s, preview GIFs and frame-mapping audits. A standard Arial or DejaVu Sans font is required.

## 4. Physical evaluation dependencies

Use Linux with a compatible NVIDIA GPU and the project's Isaac Lab / ProtoMotions environment. Install third-party simulation dependencies according to their official documentation; `requirements-media.txt` is not a simulation environment installer.

The release records observed package versions in `evidence/environment.json`. The original checkouts did not retain a Git commit identity, so this snapshot does not claim that a clean install of an arbitrary current upstream revision will reproduce the original environment.

Required assets:

| Asset | Used for |
|---|---|
| ScanNet scene exports and instance metadata | Physical room geometry and object binding |
| HUMANISE motions and aligned motion banks | Original reference trajectories |
| Licensed SMPL / SMPL-X assets | Humanoid geometry and mesh references |
| Frozen flat motion tracker | Basic tracking prior |
| Frozen terrain tracker | Terrain-correction prior |
| Original and planned motion banks + plan manifests | Goal-preserving reference contract |
| Qwen3-VL-4B-Instruct weights and proposal images | Optional fresh language parsing |

Asset hashes and feature versions are recorded in `evidence/selection.json` and `evidence/training_manifest.json`. Obtain restricted assets from their original providers.

## 5. Prepare a local configuration

The five templates in `configs/` retain the original 24-slot protocol and identity checks. They are not sufficient without the corresponding external assets.

```bash
export PROJECT_ROOT="$PWD"
export ASSET_ROOT="/path/to/recovery-assets"
export PROTOMOTIONS_ROOT="/path/to/ProtoMotions"
export ISAACLAB_ROOT="/path/to/IsaacLab"
export V4_VENDOR="$PROTOMOTIONS_ROOT"
export ISAACLAB_PYTHON="$ISAACLAB_ROOT/.venv/bin/python"
export SURFACE_TEACHER_SCRIPTS="$PROJECT_ROOT/scripts_teacher_planned"

python tools/prepare_config.py \
  --template configs/015317.template.json \
  --checkpoint models/h1-bc-ema.pt \
  --output-dir runs/office-evaluation \
  --out configs/015317.local.json
```

`PROJECT_ROOT/data/` and `PROJECT_ROOT/pretrained/` must contain the licensed/generated assets expected by the template, or the corresponding fields must be explicitly updated to their real locations. Hash checks remain enforced. Do not replace missing assets with different files while retaining the expected hashes.

The helper expands path variables, checks required files, and uses a new output directory. It does not start a physics job.

Then launch using the exact motion bank and tracker paths in that local config:

```bash
bash scripts_student_surface/launch.sh \
  configs/015317.local.json 24 \
  "/absolute/path/to/the/planned/015317.pt" \
  "/absolute/path/to/the/frozen/flat/last.ckpt"
```

## 6. Fresh language-to-physics example

The audited entry supports the fixed office task. It validates the compiled relations, instance IDs, selected reference and checkpoint before physics.

```bash
export QWEN_MODEL_PATH="/path/to/Qwen3-VL-4B-Instruct"
export RECOVERY_ASSET_ROOT="$ASSET_ROOT"

"$ISAACLAB_PYTHON" scripts_surface_demo/run_system.py \
  --root "$PROJECT_ROOT" \
  --scene-config configs/015317.local.json \
  --manifest "/path/to/proposal_manifest_v2.json" \
  --compiler-directory "$PROJECT_ROOT/scripts_demo" \
  --teacher-runtime "$PROJECT_ROOT/scripts_teacher_planned" \
  --student-runtime "$PROJECT_ROOT/scripts_student_surface" \
  --method bc --checkpoint "$PROJECT_ROOT/models/h1-bc-ema.pt" \
  --execute-steps 1 --inference-seed 17 \
  --output "$PROJECT_ROOT/runs/office-preflight" \
  --prepare-only
```

For execution, remove `--prepare-only` and choose another empty output directory. Qwen parsing completes before the physical evaluation starts. This is the fixed supported demonstration, not an arbitrary-language benchmark.

## 7. Offline student training

The training entry requires successful teacher episodes in the existing dataset format, including consistent prior and reference identities. The release does not bundle all 2,609 training trajectories.

```bash
"$ISAACLAB_PYTHON" scripts_student_surface/train_student.py \
  --data /path/to/full /path/to/takeover /path/to/late_takeover \
         /path/to/bc_early /path/to/bc_hold \
         /path/to/diffusion_early /path/to/diffusion_hold \
  --output runs/new_h1_bc \
  --kind bc --horizon 1 --steps 15000 --batch-size 256 \
  --seed 109 --sampling balanced
```

Use a new directory for each run. Training new weights does not reproduce the published physical scores until those weights have been evaluated under the same protocol.

## Publication changes

Model and task algorithms are preserved from the active runtime snapshot. Changes for this public release are declared in `evidence/source_provenance.json`: environment-based launcher paths, a configurable Qwen model location, and normalized source line endings. The English video renderer, gallery, documentation and verification utilities are new publication tooling.
