#include "cqi_udp_listener.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <pthread.h>
#include <sys/time.h>

static float global_cqi_state[96];
static pthread_mutex_t cqi_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool has_new_cqi = false;

static void* udp_listener_thread(void* arg) {
    int port = *(int*)arg;
    free(arg);

    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("UDP socket creation failed");
        return NULL;
    }
    
    memset(&server_addr, 0, sizeof(server_addr));
    memset(&client_addr, 0, sizeof(client_addr));
    
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);
    
    if (bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("UDP bind failed");
        close(sockfd);
        return NULL;
    }
    
    printf("[CQI_UDP] Listening for CQI data on port %d...\n", port);
    
    float recv_buffer[96];
    socklen_t len = sizeof(client_addr);
    
    while (1) {
        int n = recvfrom(sockfd, recv_buffer, sizeof(recv_buffer), 
                         MSG_WAITALL, (struct sockaddr *)&client_addr, &len);
        if (n == sizeof(recv_buffer)) {
            pthread_mutex_lock(&cqi_mutex);
            memcpy(global_cqi_state, recv_buffer, sizeof(global_cqi_state));
            has_new_cqi = true;
            pthread_mutex_unlock(&cqi_mutex);
        } else if (n > 0) {
            printf("[CQI_UDP] Warning: Received %d bytes, expected %lu bytes.\n", n, sizeof(recv_buffer));
        }
    }
    
    close(sockfd);
    return NULL;
}

void start_cqi_udp_listener(int port) {
    memset(global_cqi_state, 0, sizeof(global_cqi_state));
    pthread_t tid;
    int* port_ptr = malloc(sizeof(int));
    *port_ptr = port;
    if (pthread_create(&tid, NULL, udp_listener_thread, port_ptr) != 0) {
        perror("[CQI_UDP] Failed to create UDP listener thread");
    }
}

bool get_latest_cqi_state(float* state_buffer) {
    bool updated = false;
    pthread_mutex_lock(&cqi_mutex);
    if (has_new_cqi) {
        memcpy(state_buffer, global_cqi_state, sizeof(global_cqi_state));
        has_new_cqi = false;
        updated = true;
    }
    pthread_mutex_unlock(&cqi_mutex);
    return updated;
}

long long get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (((long long)tv.tv_sec)*1000000)+(tv.tv_usec);
}
