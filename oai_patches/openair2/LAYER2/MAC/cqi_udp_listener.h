#ifndef CQI_UDP_LISTENER_H
#define CQI_UDP_LISTENER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Start the UDP listener thread on the specified port
void start_cqi_udp_listener(int port);

// Get the latest CQI state vector
// state_buffer must be pre-allocated to hold 96 floats (3 RUs * 8 PRBs * 4 UEs)
// Returns true if new data was copied, false otherwise
bool get_latest_cqi_state(float* state_buffer);

// Get current timestamp in microseconds for profiling
long long get_time_us(void);

#ifdef __cplusplus
}
#endif

#endif // CQI_UDP_LISTENER_H
