#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <limits.h>

// --- Estruturas de Dados ---

typedef struct {
    int* coords;     
    int cluster_id;  
} Point;

typedef struct {
    int* coords;
} Centroid;

// Declaração antecipada
struct ThreadArgs;

typedef struct ThreadArgs {
    int thread_id;
    int num_threads;
    int start_idx;
    int end_idx;
    
    // Dados compartilhados (Leitura)
    Point* points;
    Point* centroids;
    int M, K, D, I;
    
    // Sincronização
    pthread_barrier_t* barrier;
    
    // Acumuladores Locais (privados da thread)
    long long* local_sum_coords;
    int* local_counts;
    
    // Ponteiro para o array de argumentos de TODAS as threads
    // (Para que a Thread 0 possa acessar os dados das outras)
    struct ThreadArgs* all_thread_args; 
    
} ThreadArgs;

// --- Função de Distância Otimizada (Loop Unrolling) ---
// essa otimização pois ela ajuda no processamento vetorial
long long euclidean_dist_sq(int* p1_coords, int* p2_coords, int D) {
    long long dist = 0;
    int i = 0;
    
    // Processa 4 dimensões por vez
    for (; i <= D - 4; i += 4) {
        long long d0 = (long long)p1_coords[i]   - p2_coords[i];
        long long d1 = (long long)p1_coords[i+1] - p2_coords[i+1];
        long long d2 = (long long)p1_coords[i+2] - p2_coords[i+2];
        long long d3 = (long long)p1_coords[i+3] - p2_coords[i+3];
        dist += d0*d0 + d1*d1 + d2*d2 + d3*d3;
    }
    
    // Restante
    for (; i < D; i++) {
        long long diff = (long long)p1_coords[i] - p2_coords[i];
        dist += diff * diff;
    }
    return dist;
}

void read_data_from_file(const char* filename, Point* points, int M, int D) {
    FILE* file = fopen(filename, "r");
    if (!file) { perror("Erro arquivo"); exit(EXIT_FAILURE); }
    int* all_coords = (int*)malloc(M * D * sizeof(int));
    for (int i = 0; i < M; i++) {
        points[i].coords = &all_coords[i * D];
        for (int d = 0; d < D; d++) {
            fscanf(file, "%d", &points[i].coords[d]);
        }
        points[i].cluster_id = -1;
    }
    fclose(file);
}

void initialize_centroids(Point* points, Point* centroids, int M, int K, int D) {
    srand(10); 
    int* all_centroid_coords = (int*)malloc(K * D * sizeof(int));
    for (int i = 0; i < K; i++) {
        centroids[i].coords = &all_centroid_coords[i * D];
        int idx = rand() % M;
        for (int d = 0; d < D; d++) {
            centroids[i].coords[d] = points[idx].coords[d];
        }
    }
}

// --- Worker Otimizado (SEM MUTEX) ---

void* thread_worker(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    
    // Alocação local (persiste durante toda a vida da thread)
    args->local_sum_coords = (long long*)malloc(args->K * args->D * sizeof(long long));
    args->local_counts = (int*)malloc(args->K * sizeof(int));

    for (int iter = 0; iter < args->I; iter++) {
        
        // Limpa acumuladores locais
        memset(args->local_sum_coords, 0, args->K * args->D * sizeof(long long));
        memset(args->local_counts, 0, args->K * sizeof(int));

        // Fase de Cálculo (Paralelo)
        for (int i = args->start_idx; i < args->end_idx; i++) {
            long long min_dist = LLONG_MAX;
            int best_cluster = 0;

            // Busca linear pelo cluster mais próximo
            for (int k = 0; k < args->K; k++) {
                long long dist = euclidean_dist_sq(args->points[i].coords, args->centroids[k].coords, args->D);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }
            
            args->points[i].cluster_id = best_cluster;
            args->local_counts[best_cluster]++;
            
            // Loop unrolling na acumulação local para performance extra
            int d = 0;
            int offset = best_cluster * args->D;
            for (; d <= args->D - 4; d += 4) {
                args->local_sum_coords[offset + d]     += args->points[i].coords[d];
                args->local_sum_coords[offset + d + 1] += args->points[i].coords[d+1];
                args->local_sum_coords[offset + d + 2] += args->points[i].coords[d+2];
                args->local_sum_coords[offset + d + 3] += args->points[i].coords[d+3];
            }
            for (; d < args->D; d++) {
                args->local_sum_coords[offset + d] += args->points[i].coords[d];
            }
        }

        // Barreira: Espera todos terminarem de calcular
        pthread_barrier_wait(args->barrier);

        // Fase de Redução e Atualização (Apenas Thread 0)
        // Como todos estão parados na barreira (ou esperando na próxima),
        // A thread 0 tem acesso exclusivo de leitura aos dados locais e escrita nos centroides.
        if (args->thread_id == 0) {
            
            // Para cada cluster
            for (int k = 0; k < args->K; k++) {
                long long total_sum[args->D]; // Buffer temporário para soma
                memset(total_sum, 0, args->D * sizeof(long long));
                int total_count = 0;

                // Agrega resultados de TODAS as threads (Redução)
                for (int t = 0; t < args->num_threads; t++) {
                    ThreadArgs* t_data = &args->all_thread_args[t];
                    total_count += t_data->local_counts[k];
                    
                    for (int d = 0; d < args->D; d++) {
                        total_sum[d] += t_data->local_sum_coords[k * args->D + d];
                    }
                }

                // Atualiza o centroide global
                if (total_count > 0) {
                    for (int d = 0; d < args->D; d++) {
                        args->centroids[k].coords[d] = total_sum[d] / total_count;
                    }
                }
            }
        }

        // Barreira: Espera Thread 0 atualizar os centroides antes de começar a próxima iter
        pthread_barrier_wait(args->barrier);
    }

    free(args->local_sum_coords);
    free(args->local_counts);
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    if (argc < 6) return EXIT_FAILURE;

    const char* filename = argv[1];
    int M = atoi(argv[2]);
    int D = atoi(argv[3]);
    int K = atoi(argv[4]);
    int I = atoi(argv[5]);
    
    int num_threads = 4;
    char* env_threads = getenv("NUM_THREADS");
    if (env_threads) num_threads = atoi(env_threads);

    Point* points = (Point*)malloc(M * sizeof(Point));
    Point* centroids = (Point*)malloc(K * sizeof(Point));
    
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, num_threads);

    read_data_from_file(filename, points, M, D);
    initialize_centroids(points, centroids, M, K, D);

    // Estruturas de Threads
    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadArgs* t_args = (ThreadArgs*)malloc(num_threads * sizeof(ThreadArgs));
    
    int points_per_thread = M / num_threads;
    int remainder = M % num_threads;
    int current_idx = 0;

    // Configuração
    for (int i = 0; i < num_threads; i++) {
        t_args[i].thread_id = i;
        t_args[i].num_threads = num_threads;
        t_args[i].points = points;
        t_args[i].centroids = centroids; // Todos leem os mesmos centroides
        t_args[i].M = M; t_args[i].K = K; t_args[i].D = D; t_args[i].I = I;
        t_args[i].barrier = &barrier;
        t_args[i].all_thread_args = t_args; // Passa o vetor de structs para todos

        int count = points_per_thread + (i < remainder ? 1 : 0);
        t_args[i].start_idx = current_idx;
        t_args[i].end_idx = current_idx + count;
        current_idx += count;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, thread_worker, (void*)&t_args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    
    printf("Tempo total: %.5f segundos\n", time_taken);
    

    pthread_barrier_destroy(&barrier);
    free(threads);
    free(t_args);

    return EXIT_SUCCESS;
}