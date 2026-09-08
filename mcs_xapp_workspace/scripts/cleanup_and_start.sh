#!/bin/bash
pkill -9 -f toggle_physical_snr.py
pkill -9 -f plot_live.py
pkill -9 -f send_cdl.py
pkill -9 -f run_drl.sh
pkill -9 -f traffic_gen.py
echo 'admin@1234' | sudo -S pkill -9 nr-softmodem
echo 'admin@1234' | sudo -S pkill -9 nr-uesoftmodem
sleep 2
./run_drl_sim.sh
