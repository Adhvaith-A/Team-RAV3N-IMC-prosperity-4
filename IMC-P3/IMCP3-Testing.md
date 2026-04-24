# IMC Prosperity 3 Testing

Setup And Testing-Training for a IMCP3 Workflow

## Scope

- Strategy file for IMCP3: `trader_imcp3_r2.py`
- Canonical IMCP3 data source: bundled resources from `prosperity3bt`

## 1) Environment Setup

```bash
cd /Users/<user-name>/Directory/RAV3N
python3 -m venv .venv-imcp3
source .venv-imcp3/bin/activate
python -m pip install -U pip
pip install -U prosperity3bt
```

Verify install:

```bash
prosperity3bt --help
```

## 2) First Smoke Run

Run a default round/day preset via CLI:

```bash
mkdir -p runs/imcp3
prosperity3bt trader_imcp3_r2.py 0 --out runs/imcp3/r0.log --print
```

If your local CLI build uses explicit round/day flags, use:

```bash
prosperity3bt trader_imcp3_r2.py --round 1 --day 0 --out runs/imcp3/r1d0.log --print
```

## 3) Visualizer

Hosted visualizer:

- https://jmerle.github.io/imc-prosperity-3-visualizer/

Use generated log files from `runs/imcp3/`.

Direct launch from CLI (if supported by installed version):

```bash
prosperity3bt trader_imcp3_r2.py 0 --out runs/imcp3/r0.log --vis
```

## 4) Optional Local Tool Copies

Clone tools into sibling folders:

```bash
cd /Users/<user-name>/Documents
git clone https://github.com/jmerle/imc-prosperity-3-backtester.git prosperity3-tools-backtester
git clone https://github.com/jmerle/imc-prosperity-3-visualizer.git prosperity3-tools-visualizer
```

Visualizer local dev:

```bash
cd /Users/<user-name>/Direcoty/prosperity3-tools-visualizer
corepack enable
pnpm install
pnpm dev
```

## 5) Data Provenance And Integrity

Primary source is `prosperity3bt` bundled resources.

Inspect resource location in cloned repo:

```bash
ls /Users/<user-name>/Directory/prosperity3-tools-backtester/prosperity3bt/resources
```

Checksum workflow to verify a mirror against canonical resources:

```bash
find /path/to/canonical/resources -type f -name "*.csv" -print0 | xargs -0 shasum -a 256 | sort > canonical.sha
find /path/to/mirror/resources -type f -name "*.csv" -print0 | xargs -0 shasum -a 256 | sort > mirror.sha
diff -u canonical.sha mirror.sha
```

