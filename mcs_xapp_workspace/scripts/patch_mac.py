import sys

filename = "/home/ngn-6g/OAI/openairinterface5g/openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_dlsch.c"
with open(filename, "r") as f:
    text = f.read()

old_str = """      } else
        selected_mcs = get_mcs_from_bler(bo, stats, &sched_ctrl->dl_bler_stats, max_mcs, frame);
      int l = get_dl_nrOfLayers(sched_ctrl, current_BWP->dci_format);"""

new_str = """      } else
        selected_mcs = get_mcs_from_bler(bo, stats, &sched_ctrl->dl_bler_stats, max_mcs, frame);

      // 3) Apply PA distortion CQI penalty from agent (per-UE distortion).
      selected_mcs = max(0, selected_mcs - (int)UE->cqi_downgrade);

      int l = get_dl_nrOfLayers(sched_ctrl, current_BWP->dci_format);"""

if old_str in text:
    text = text.replace(old_str, new_str)
    with open(filename, "w") as f:
        f.write(text)
    print("Patched MAC successfully.")
else:
    print("Could not find target string in MAC.")
