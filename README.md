# MCS Selection O-RAN xApp — OAI Integration

> **Deep Reinforcement Learning-based Modulation and Coding Scheme (MCS) selection for 5G NR gNB, integrated into the OpenAirInterface MAC scheduler via ONNX Runtime.**

---

## Overview

This repository contains **only the project-specific files** required to integrate the DRL-based MCS xApp into the [OpenAirInterface (OAI)](https://gitlab.eurecom.fr/oai/openairinterface5g) 5G NR stack. The full OAI codebase is **not included** — you clone OAI separately and patch in these files.

The agent runs at the gNB MAC layer, intercepting MCS decisions in the DLSCH scheduler and replacing them with inference from a trained ONNX model. It uses SRS channel estimates (real + imaginary), per-UE SNR, previous MCS, HARQ feedback, and a distortion proxy (`alpha`) as inputs.

---

## Repository Structure

```
mcs-selection-xapp/
│
├── README.md
│
├── oai_patches/             ← Modified OAI C/C++ source files (drop-in replacements)
│   ├── openair1/
│   │   ├── PHY/INIT/nr_init.c
│   │   ├── PHY/INIT/nr_init_ru.c
│   │   ├── PHY/NR_TRANSPORT/nr_dlsch.c
│   │   ├── PHY/NR_TRANSPORT/pucch_rx.c
│   │   ├── PHY/NR_UE_ESTIMATION/nr_ue_measurements.c
│   │   ├── PHY/defs_gNB.h
│   │   ├── SCHED_NR/phy_procedures_nr_gNB.c
│   │   └── SIMULATION/TOOLS/{multipath_channel,random_channel}.c
│   ├── openair2/
│   │   ├── LAYER2/MAC/cqi_udp_listener.{c,h}   ← NEW
│   │   ├── LAYER2/MAC/pre_processor.c
│   │   ├── LAYER2/MAC/slicing/slicing.c
│   │   ├── LAYER2/MAC/torch_infer.{cpp,h}       ← NEW
│   │   ├── LAYER2/NR_MAC_UE/nr_ra_procedures.c
│   │   ├── LAYER2/NR_MAC_UE/nr_ue_power_procedures.c
│   │   ├── LAYER2/NR_MAC_gNB/gNB_scheduler*.c
│   │   ├── LAYER2/NR_MAC_gNB/onnx_agent.{c,h}   ← NEW: ONNX inference engine
│   │   ├── LAYER2/NR_MAC_gNB/nr_mac_gNB.h
│   │   └── RRC/NR_UE/rrc_timers_and_constants.c
│   ├── radio/
│   │   ├── COMMON/common_lib.{c,h}
│   │   └── rfsimulator/{apply_channelmod.c,rfsimulator.h,simulator.cpp}
│   └── nfapi/open-nFAPI/fapi/src/nr_fapi_p7.c
│
├── configs/                 ← gNB/UE config files for RF simulator
│   ├── gnb_drl_agent.conf              ← Main 32-antenna DRL gNB config
│   ├── gnb.sa.band78.fr1.51PRB.drl.conf
│   ├── gnb.sa.band78.fr1.106PRB.num2.conf
│   ├── gnb.sa.band78.fr1.24PRB.num2.conf
│   ├── channelmod_rfsimu.conf
│   ├── channelmod_tdlc.conf
│   └── ue.conf
│
├── mcs_xapp_workspace/
│   ├── models/
│   │   ├── AgentNetwork.onnx
│   │   ├── GSM_D3QN_online.onnx        ← Primary deployed model
│   │   └── GSM_D3QN_theo_online.onnx
│   ├── scripts/
│   │   ├── run_drl_sim.sh              ← Main entry point
│   │   ├── run_drl_agent.py
│   │   ├── run_baseline.py
│   │   ├── collect_telemetry.py
│   │   ├── plot_live.py
│   │   ├── sweep_snr_gsm.py
│   │   ├── toggle_physical_snr.py
│   │   └── test_onnx_*.py
│   └── docs/
│       └── OAI_implementation_points.txt
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Ubuntu | 22.04 LTS |
| CMake | ≥ 3.16 |
| GCC | ≥ 11 |
| Python | 3.10+ |
| ONNX Runtime | 1.16.3 |

---

## Integration Steps

### 1. Clone OAI (tag 2026.w35)

```bash
git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git openairinterface5g
cd openairinterface5g
git checkout 2026.w35
```

### 2. Build OAI (Base Stack)

```bash
source oaienv
cd cmake_targets
./build_oai -I          # Install system deps (first time only)
./build_oai --gNB --nrUE -w SIMU --build-lib "telnetsrv" -C
```

### 3. Clone This Repository

```bash
git clone https://github.com/<your-org>/mcs-selection-xapp.git
cd mcs-selection-xapp
```

### 4. Install ONNX Runtime

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
mv onnxruntime-linux-x64-1.16.3 /path/to/openairinterface5g/onnxruntime
```

### 5. Copy Patched Source Files

From the OAI root directory:

```bash
cp -r oai_patches/openair1/* openair1/
cp -r oai_patches/openair2/* openair2/
cp -r oai_patches/radio/*    radio/
cp -r oai_patches/nfapi/*    nfapi/
```

### 6. Copy Config Files

```bash
cp configs/gnb_drl_agent.conf .
cp configs/gnb.sa.band78.fr1.51PRB.drl.conf  targets/PROJECTS/GENERIC-NR-5GC/CONF/
cp configs/channelmod_rfsimu.conf             targets/PROJECTS/GENERIC-NR-5GC/CONF/
cp configs/ue.conf                            targets/PROJECTS/GENERIC-NR-5GC/CONF/
```

### 7. Rebuild gNB

```bash
cd cmake_targets
./build_oai --gNB --nrUE -w SIMU --build-lib "telnetsrv" -C
```

> **Note:** If the linker cannot find `libonnxruntime.so`:
> ```bash
> export LD_LIBRARY_PATH=/path/to/openairinterface5g/onnxruntime/lib:$LD_LIBRARY_PATH
> ```

### 8. Copy ONNX Models

```bash
cp mcs_xapp_workspace/models/*.onnx /path/to/openairinterface5g/
```

---

## Running the Experiment

### DRL Agent Mode

```bash
cd mcs_xapp_workspace/scripts
./run_drl_sim.sh
```

This script starts the gNB (ONNX agent active), the RF simulator UE, and the telemetry collector.

### Baseline Mode

```bash
python3 mcs_xapp_workspace/scripts/run_baseline.py
```

### Live Monitoring

```bash
python3 mcs_xapp_workspace/scripts/plot_live.py
```

## Key Modified Files

| File | Nature of Change |
|---|---|
| `openair2/LAYER2/NR_MAC_gNB/onnx_agent.c` | **NEW** ONNX Runtime inference wrapper |
| `openair2/LAYER2/NR_MAC_gNB/onnx_agent.h` | **NEW** C API: `init_onnx_agent`, `run_onnx_agent` |
| `openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_dlsch.c` | Hook: replaces table MCS with agent output |
| `openair2/LAYER2/NR_MAC_gNB/nr_mac_gNB.h` | Adds `onnx_initialized`, `prev_mcs`, `alpha` fields |
| `openair2/LAYER2/MAC/cqi_udp_listener.c` | **NEW** UDP MAC statistics broadcaster |
| `openair1/PHY/INIT/nr_init.c` | SRS buffer for 32-antenna MIMO |
| `openair1/PHY/NR_UE_ESTIMATION/nr_ue_measurements.c` | Exposes SRS H estimates to MAC |
| `radio/rfsimulator/simulator.cpp` | CDL-C taps + SNR telnet control |

---


## License

Project-specific files (ONNX agent, scripts, configs): **MIT License**  
Modified OAI source files: **OAI Public License v1.1** — see [OAI LICENSE](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/LICENSE)
