with open("radio/rfsimulator/simulator.cpp", "r") as f:
    content = f.read()

target = """          sample_t *in = (sample_t *)samplesVoid[tx_beams[beam]][a];
          if (in == NULL || nsamps <= 0 || wsize <= 0) {
            LOG_E(HW, "BAD POINTER OR SIZE: in=%p, nsamps=%d, wsize=%zd\\n", in, nsamps, wsize);
          }
          if (flags == 2)
            b->th.nbAnt = 0; // fake number of antennas to indicate a sidelink packet
          int ret = fullwrite(buf->conn_sock, (void *)in, wsize, t);
          if (ret != (int)wsize)
            LOG_W(HW, "rfsimulator failed to write to socket\\n");"""

replacement = """          sample_t *in = (sample_t *)samplesVoid[tx_beams[beam]][a];
          if (in == NULL || nsamps <= 0 || wsize <= 0) {
            LOG_E(HW, "BAD POINTER OR SIZE: in=%p, nsamps=%d, wsize=%zd. Sending zeroes to preserve TCP alignment.\\n", in, nsamps, wsize);
            void *zero_buf = calloc(1, wsize);
            if (flags == 2)
              b->th.nbAnt = 0;
            int ret = fullwrite(buf->conn_sock, zero_buf, wsize, t);
            if (ret != (int)wsize)
              LOG_W(HW, "rfsimulator failed to write dummy zeroes\\n");
            free(zero_buf);
          } else {
            if (flags == 2)
              b->th.nbAnt = 0; // fake number of antennas to indicate a sidelink packet
            int ret = fullwrite(buf->conn_sock, (void *)in, wsize, t);
            if (ret != (int)wsize)
              LOG_W(HW, "rfsimulator failed to write to socket\\n");
          }"""

if target in content:
    with open("radio/rfsimulator/simulator.cpp", "w") as f:
        f.write(content.replace(target, replacement))
    print("Patched successfully")
else:
    print("Target not found")
