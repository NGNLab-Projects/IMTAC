import sys

filename = "/home/ngn-6g/OAI/openairinterface5g/openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_dlsch.c"
with open(filename, "r") as f:
    text = f.read()

old_str = """      uint32_t tbs = nr_compute_tbs(Qm,
                                    R,
                                    n_rb_sched[beam.idx], /* rbSize */
                                    14, /* hypothetical number of slots */
                                    0, /* N_PRB_DMRS * N_DMRS_SLOT */
                                    0 /* N_PRB_oh, 0 for initialBWP */,
                                    0 /* tb_scaling */,
                                    l) >> 3;
      if (frame % 100 == 0) {
          LOG_I(NR_MAC, "UE %04x: frame %d slot %d, selected_mcs %d (cqi_downgrade %d), Qm %d, R %d, n_rb_sched %d, tbs %d bytes\\n",
                UE->rnti, frame, slot, selected_mcs, (int)UE->cqi_downgrade, Qm, R, n_rb_sched[beam.idx], tbs);
      }"""

new_str = """      uint32_t tbs = nr_compute_tbs(Qm,
                                    R,
                                    1, /* rbSize */
                                    10, /* hypothetical number of slots */
                                    0, /* N_PRB_DMRS * N_DMRS_SLOT */
                                    0 /* N_PRB_oh, 0 for initialBWP */,
                                    0 /* tb_scaling */,
                                    l) >> 3;"""

if old_str in text:
    text = text.replace(old_str, new_str)
    with open(filename, "w") as f:
        f.write(text)
    print("Patched MAC successfully.")
else:
    print("Could not find target string in MAC.")
