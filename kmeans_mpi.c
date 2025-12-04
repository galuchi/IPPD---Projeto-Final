#define _POSIX_C_SOURCE 199309L  // Necessário para CLOCK_MONOTONIC
#include <limits.h>              // Para LLONG_MAX
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>  // Header correto para clock_gettime e struct timespec
#include <math.h>
#include <mpi.h>

// Estrutura para representar um ponto no espaço D-dimensional
typedef struct {
  int* coords;     // Vetor de coordenadas inteiras
  int cluster_id;  // ID do cluster ao qual o ponto pertence
} Point;

// --- Funções Utilitárias ---

/**
 * @brief Calcula a distância Euclidiana ao quadrado entre dois pontos com coordenadas inteiras.
 * Usa 'long long' para evitar overflow no cálculo da distância e da diferença.
 * @return A distância Euclidiana ao quadrado como um long long.
 */
long long euclidean_dist_sq(Point* p1, Point* p2, int D) {
  long long dist = 0;
  for (int i = 0; i < D; i++) {
    long long diff = (long long)p1->coords[i] - p2->coords[i];
    dist += diff * diff;
  }
  return dist;
}

// --- Funções Principais do K-Means ---

/**
 * @brief Lê os dados de pontos (inteiros) de um arquivo de texto.
 */
void read_data_from_file(const char* filename, Point* points, int M, int D) {
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "Erro: Não foi possível abrir o arquivo '%s'\n", filename);
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < D; j++) {
      if (fscanf(file, "%d", &points[i].coords[j]) != 1) {
        fprintf(stderr, "Erro: Arquivo de dados mal formatado ou incompleto.\n");
        fclose(file);
        exit(EXIT_FAILURE);
      }
    }
  }

  fclose(file);
}

/**
 * @brief Inicializa os centroides escolhendo K pontos aleatórios do dataset.
 */
void initialize_centroids(Point* points, Point* centroids, int M, int K, int D) {
  srand(42);  // Semente fixa para reprodutibilidade

  int* indices = (int*)malloc(M * sizeof(int));
  for (int i = 0; i < M; i++) {
    indices[i] = i;
  }

  for (int i = 0; i < M; i++) {
    int j = rand() % M;
    int temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }

  for (int i = 0; i < K; i++) {
    memcpy(centroids[i].coords, points[indices[i]].coords, D * sizeof(int));
  }

  free(indices);
}

/**
 * @brief Fase de Atribuição: Associa cada ponto ao cluster do centroide mais próximo.
 */
void assign_points_to_clusters(Point* points, Point* centroids, int M, int K, int D) {
  for (int i = 0; i < M; i++) {
    long long min_dist = LLONG_MAX;
    int best_cluster = -1;

    for (int j = 0; j < K; j++) {
      long long dist = euclidean_dist_sq(&points[i], &centroids[j], D);
      if (dist < min_dist) {
        min_dist = dist;
        best_cluster = j;
      }
    }
    points[i].cluster_id = best_cluster;
  }
}

/**
 * @brief Fase de Atualização: Recalcula a posição de cada centroide como a média
 * (usando divisão inteira) de todos os pontos atribuídos ao seu cluster.
 */
void update_centroids(Point* points, Point* centroids, int M, int K, int D) {
  long long* cluster_sums = (long long*)calloc(K * D, sizeof(long long));
  int* cluster_counts = (int*)calloc(K, sizeof(int));

  for (int i = 0; i < M; i++) {
    int cluster_id = points[i].cluster_id;
    cluster_counts[cluster_id]++;
    for (int j = 0; j < D; j++) {
      cluster_sums[cluster_id * D + j] += points[i].coords[j];
    }
  }

  for (int i = 0; i < K; i++) {
    if (cluster_counts[i] > 0) {
      for (int j = 0; j < D; j++) {
        // Divisão inteira para manter os centroides em coordenadas discretas
        centroids[i].coords[j] = cluster_sums[i * D + j] / cluster_counts[i];
      }
    }
  }

  free(cluster_sums);
  free(cluster_counts);
}

/**
 * @brief Imprime os resultados finais e o checksum (como long long).
 */
void print_results(Point* centroids, int K, int D) {
  printf("--- Centroides Finais ---\n");
  long long checksum = 0;
  for (int i = 0; i < K; i++) {
    printf("Centroide %d: [", i);
    for (int j = 0; j < D; j++) {
      printf("%d", centroids[i].coords[j]);
      if (j < D - 1) printf(", ");
      checksum += centroids[i].coords[j];
    }
    printf("]\n");
  }
  printf("\n--- Checksum ---\n");
  printf("%lld\n", checksum);  // %lld para long long int
}

/**
 * @brief Calcula e imprime o tempo de execução e o checksum final.
 * A saída é formatada para ser facilmente lida por scripts:
 * Linha 1: Tempo de execução em segundos (double)
 * Linha 2: Checksum final (long long)
 */
void print_time_and_checksum(Point* centroids, int K, int D, double exec_time) {
  long long checksum = 0;
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < D; j++) {
      checksum += centroids[i].coords[j];
    }
  }
  // Saída formatada para o avaliador
  printf("%lf\n", exec_time);
  printf("%lld\n", checksum);
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M, D, K, I;

    // Ponteiros globais (Rank 0)
    int* all_coords = NULL;
    Point* points_global = NULL;

    // 1. LEITURA E PREPARAÇÃO (RANK 0)
    if (rank == 0) {
        if (argc != 6) MPI_Abort(MPI_COMM_WORLD, 1);

        M = atoi(argv[2]);
        D = atoi(argv[3]);
        K = atoi(argv[4]);
        I = atoi(argv[5]);

        if (M % size != 0) {
            fprintf(stderr, "Erro: M deve ser divisível por %d\n", size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        all_coords = (int*)malloc(M * D * sizeof(int));
        points_global = (Point*)malloc(M * sizeof(Point));
        for (int i = 0; i < M; i++) points_global[i].coords = &all_coords[i * D];

        read_data_from_file(argv[1], points_global, M, D);
    }

    // BROADCAST DAS DIMENSÕES
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&I, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ALOCAÇÃO LOCAL E SCATTER
    int M_local = M / size;
    int* local_coords = (int*)malloc(M_local * D * sizeof(int));
    Point* local_points = (Point*)malloc(M_local * sizeof(Point));

    // Configura structs locais
    for (int i = 0; i < M_local; i++) {
        local_points[i].coords = &local_coords[i * D];
        local_points[i].cluster_id = -1;
    }

    // Distribui os pontos
    MPI_Scatter(all_coords, M_local * D, MPI_INT, 
                local_coords, M_local * D, MPI_INT, 
                0, MPI_COMM_WORLD);

    // PREPARAÇÃO DOS CENTRÓIDES
    // Vetor linear para armazenar/comunicar centróides (K * D inteiros)
    int* centroids_coords = (int*)malloc(K * D * sizeof(int));
    Point* centroids = (Point*)malloc(K * sizeof(Point));
    for (int k = 0; k < K; k++) centroids[k].coords = &centroids_coords[k * D];

    // Rank 0 inicializa os centróides aleatoriamente na primeira vez
    if (rank == 0) {
        initialize_centroids(points_global, centroids, M, K, D);
    }

    // Buffers para redução (Soma das coordenadas e contagem de pontos por cluster)
    long long* local_sums = (long long*)malloc(K * D * sizeof(long long));
    int* local_counts = (int*)malloc(K * sizeof(int));

    long long* global_sums = NULL;
    int* global_counts = NULL;

    if (rank == 0) {
        global_sums = (long long*)malloc(K * D * sizeof(long long));
        global_counts = (int*)malloc(K * sizeof(int));
    }

    // --- LOOP PRINCIPAL DO K-MEANS ---
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int iter = 0; iter < I; iter++) {
        // A. Rank 0 envia os centróides atuais para todos
        MPI_Bcast(centroids_coords, K * D, MPI_INT, 0, MPI_COMM_WORLD);

        // B. Zerar acumuladores locais
        memset(local_sums, 0, K * D * sizeof(long long));
        memset(local_counts, 0, K * sizeof(int));

        // C. Fase de Atribuição (Paralela)
        for (int i = 0; i < M_local; i++) {
            long long min_dist = LLONG_MAX;
            int best_cluster = -1;

            for (int k = 0; k < K; k++) {
                long long dist = euclidean_dist_sq(&local_points[i], &centroids[k], D);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }
            local_points[i].cluster_id = best_cluster;

            // Acumula para a fase de atualização
            local_counts[best_cluster]++;
            for (int d = 0; d < D; d++) {
                local_sums[best_cluster * D + d] += local_points[i].coords[d];
            }
        }

        // D. Redução (Soma dos parciais no Rank 0)
        MPI_Reduce(local_sums, global_sums, K * D, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_counts, global_counts, K, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // E. Atualização dos Centróides (Apenas Rank 0)
        if (rank == 0) {
            for (int k = 0; k < K; k++) {
                if (global_counts[k] > 0) {
                    for (int d = 0; d < D; d++) {
                        centroids[k].coords[d] = global_sums[k * D + d] / global_counts[k];
                    }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // --- RESULTADOS FINAIS (RANK 0) ---
    if (rank == 0) {
        long long checksum = 0;
        for (int k = 0; k < K; k++) {
            for (int d = 0; d < D; d++) {
                checksum += centroids[k].coords[d];
            }
        }
        print_time_and_checksum(centroids, K, D, end_time - start_time);

        free(all_coords); free(points_global);
        free(global_sums); free(global_counts);
    }

    // Limpeza geral
    free(local_coords); free(local_points);
    free(centroids_coords); free(centroids);
    free(local_sums); free(local_counts);

    MPI_Finalize();
    return EXIT_SUCCESS;
}