#!/bin/bash

trap 'echo "Caught Ctrl+C, cleaning up..."; pkill -P $$; exit 1' SIGINT

echo "Cleaning up old processes..."
pkill -f cdl_c_generator.py
pkill -f plot_live.py
pkill -f toggle_physical_snr.py

echo "Starting CDL-C Generator in the background..."
python3 /home/ngn-6g/OAI/openairinterface5g1/cdl_c_generator.py &

echo "Starting Live Dashboard in the background..."
python3 /home/ngn-6g/OAI/openairinterface5g1/plot_live.py &

echo "Waiting 20 seconds for gNB and UE RRC setup..."
sleep 20

echo "Starting Ping traffic..."
python3 /home/ngn-6g/OAI/openairinterface5g1/traffic_gen.py &

echo "Starting SNR Toggler..."
python3 /home/ngn-6g/OAI/openairinterface5g1/toggle_physical_snr.py &

echo "All services running. Press Ctrl+C to stop."
wait
