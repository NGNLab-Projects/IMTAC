import os
import shutil

src = "/home/ngn-6g/OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.24PRB.usrpb210.conf"
dst = "/home/ngn-6g/OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.24PRB.num2.conf"

with open(src, "r") as f:
    text = f.read()

text = text.replace("dl_subcarrierSpacing                                           = 1;", "dl_subcarrierSpacing                                           = 2;")
text = text.replace("initialDLBWPsubcarrierSpacing = 1;", "initialDLBWPsubcarrierSpacing = 2;")
text = text.replace("ul_subcarrierSpacing = 1;", "ul_subcarrierSpacing = 2;")
text = text.replace("initialULBWPsubcarrierSpacing = 1;", "initialULBWPsubcarrierSpacing = 2;")
text = text.replace("subcarrierSpacing                                             = 1;", "subcarrierSpacing                                             = 2;")

# Also fix the CCE bottleneck by using index 12 (3 symbols) for 24 PRBs
text = text.replace("initialDLBWPcontrolResourceSetZero                              = 2;", "initialDLBWPcontrolResourceSetZero                              = 12;")

with open(dst, "w") as f:
    f.write(text)
print("Created num2 conf")
