#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define ORDER 500

int main(int argc, char** argv) {
    int rank, numProcesses;
    int blockSize, numBlocks;
    int i, j, k;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    blockSize = ORDER / sqrt(numProcesses);
    numBlocks = ORDER / blockSize;

    // Create buffers for local blocks of matrices A, B, and C
    int* localA = (int*)malloc(blockSize * blockSize * sizeof(int));
    int* localB = (int*)malloc(blockSize * blockSize * sizeof(int));
    int* localC = (int*)malloc(blockSize * blockSize * sizeof(int));

    if (rank == 0) {
        // Generate matrices A and B
        int* A = (int*)malloc(ORDER * ORDER * sizeof(int));
        int* B = (int*)malloc(ORDER * ORDER * sizeof(int));
        int* C = (int*)malloc(ORDER * ORDER * sizeof(int));

        // Initialize matrices A and B
        for (i = 0; i < ORDER; i++) {
            for (j = 0; j < ORDER; j++) {
                A[i * ORDER + j] = rand() % 100; // Random value between 0 and 100
                B[i * ORDER + j] = rand() % 100; // Random value between 0 and 100
            }
        }

        // Scatter blocks of matrices A and B to other processes
        for (i = 1; i < numProcesses; i++) {
            int row = (i - 1) / numBlocks;
            int col = (i - 1) % numBlocks;
            int* blockA = A + (row * blockSize * ORDER) + (col * blockSize);
            int* blockB = B + (row * blockSize * ORDER) + (col * blockSize);
            MPI_Send(blockA, blockSize * blockSize, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(blockB, blockSize * blockSize, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        // Calculate C1,1
        for (i = 0; i < blockSize; i++) {
            for (j = 0; j < blockSize; j++) {
                C[i * ORDER + j] = A[i * ORDER + j] + B[i * ORDER + j];
                printf("C[%d][%d] = %d\n", i, j, C[i * ORDER + j]);
            }
        }
        

        // Gather blocks of matrix C from other processes
        for (i = 1; i < numProcesses; i++) {
            int row = (i - 1) / numBlocks;
            int col = (i - 1) % numBlocks;
            int* blockC = C + (row * blockSize * ORDER) + (col * blockSize);
            MPI_Recv(blockC, blockSize * blockSize, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Print matrix C
        // printf("Matrix C:\n");
        // for (i = 0; i < ORDER; i++) {
        //     for (j = 0; j < ORDER; j++) {
        //         printf("%d ", C[i * ORDER + j]);
        //     }
        //     printf("\n");
        // }

        free(A);
        free(B);
        free(C);
    } else {
        // Receive blocks of matrices A and B from process 0
        MPI_Recv(localA, blockSize * blockSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(localB, blockSize * blockSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Perform local matrix addition
        for (i = 0; i < blockSize; i++) {
            for (j = 0; j < blockSize; j++) {
                localC[i * blockSize + j] = localA[i * blockSize + j] + localB[i * blockSize + j];
                printf("localC[%d][%d] = %d\n", i, j, localC[i * blockSize + j]);
            }
        }
        printf("\n");

        // Send the local block of matrix C to process 0
        MPI_Send(localC, blockSize * blockSize, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    free(localA);
    free(localB);
    free(localC);

    MPI_Finalize();

    return 0;
}
