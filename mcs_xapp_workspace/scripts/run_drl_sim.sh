#!/bin/bash

# Navigate to the build directory
BUILD_DIR="/home/ngn-6g/OAI/openairinterface5g1/cmake_targets/ran_build/build"
CONF_FILE="../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.51PRB.drl.conf"

echo "Starting OAI gNB in RF Simulator mode (Background)..."
echo 'admin@1234' | sudo -S bash -c "cd $BUILD_DIR; ./nr-softmodem -O $CONF_FILE --rfsim --rfsimulator.options chanmod --telnetsrv --channelmod.modellist modellist_rfsimu_1 --sa --noS1" > /tmp/gnb.log 2>&1 &

sleep 3 # Give the gNB a few seconds to initialize its socket

echo "Starting OAI UE in RF Simulator mode (Background)..."
echo 'admin@1234' | sudo -S bash -c "cd $BUILD_DIR; ./nr-uesoftmodem -O ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/channelmod_tdlc.conf -r 51 --numerology 1 --band 78 -C 3550080000 --ssb 186 --ue-nb-ant-rx 4 --ue-nb-ant-tx 4 --rfsim --rfsimulator.options chanmod --telnetsrv --telnetsrv.listenport 9091 --sa --noS1" > /tmp/ue.log 2>&1 &

echo "gNB and UE launched successfully in the background! Logs are in /tmp/gnb.log and /tmp/ue.log"
