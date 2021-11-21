#include<string.h>
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include <omp.h>
#include"mpi.h"

#define TAG 13
//Matrix N X N
#define N 3
//Número de Threads para paralelização
#define NUMTHREADS 2
//Para debug
#define DEBUG

int main(int argc, char **argv){
    int *A,*Aoriginal,*B,*C,*Cfinal;
    int numProcs,myRank,cargaTrabalho;
    int numThreads = NUMTHREADS;
    int i,j,k;
    double startTime, endTime;
    MPI_Comm USED_PROCESS;

    //Inicializa o MPI e pega o número de processos e o rank do processo atual
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    //Define o quanto cada processo irá receber
    if(numProcs >= N)
        cargaTrabalho = 1;
    else
        cargaTrabalho = N/numProcs;
    
    //Seta o número de threads do OMP
    omp_set_num_threads(numThreads);
    
    //Seta a seed para o rand
    srand(2021);

    //Coloca os processos que serão usados em um sub-comunicador com cor 1 e os processo que não irão ser utilizados em outro com cor 0
    MPI_Comm_split(MPI_COMM_WORLD, (myRank<N), myRank, &USED_PROCESS);

    //Processos alocam as matrizes que irão utilizar nos calculos
    if(myRank < N){
        //Aloca a matriz A
        A = (int *) malloc(sizeof(int) * N * cargaTrabalho);
        //Aloca a matriz B
        B = (int *) malloc(sizeof(int) * N * N);
        //Aloca a matriz C
        C = (int *) calloc(N * cargaTrabalho,sizeof(int));
    }

    //Processo 0 aloca as matrizes originais (a matriz B já foi alocada)
    if(myRank == 0){
        //Aloca a matriz A original
        Aoriginal = (int *) malloc (sizeof(int) * N * N);
        //Aloca a matriz C Final
        Cfinal = (int *) calloc (N*N,sizeof(int));

        //Preenche Aoriginal e B (NÃO TÁ ALEÁTORIO LÁ DO JEITO QUE TEM SER)
        for(i = 0; i < N; i++){
            for(j = 0; j < N; j++){
                //Aoriginal[i][j] = k;
                //B[i][j] = k;
                Aoriginal[i*N+j] = rand();
                B[i*N+j] = rand();
            }
        }

        #ifdef DEBUG
        printf("Aoriginal: \n");
        for(i = 0; i < N; i++){
            for(j = 0; j < N; j++){
                printf("%d ",Aoriginal[i*N+j]);
            }
            printf("\n");
        }

        printf("B: \n");
        for(i = 0; i < N; i++){
            for(j = 0; j < N; j++){
                printf("%d ",B[i*N+j]);
            }
            printf("\n");
        }

        printf("\n");
        #endif
    }

    
    if(myRank < N){
        //Manda um pedaço da matriz Aoriginal equivalente a carga de trabalho para cada processo
        if(MPI_Scatter(Aoriginal, N*cargaTrabalho, MPI_INT, A, N*cargaTrabalho, MPI_INT, 0, USED_PROCESS) != MPI_SUCCESS){
            printf("SCATTER ERROR\n");
            exit(1);
        }
        //Manda a matrix B inteira para todos processos
        MPI_Bcast(B,N*N,MPI_INT,0,USED_PROCESS);

        //Começa o timer
        if (myRank == 0) {
            startTime = omp_get_wtime();
        }

        //Cada processo faz seus calculos paralelamente
        #pragma omp parallel for collapse(2)
        for (i = 0; i < cargaTrabalho; i++) {
            for (j = 0; j < N; j++) {
                int aux = 0;

                #ifdef DEBUG
                    printf("On thread %d, i = %d, j = %d\n", omp_get_thread_num(), i, j);
                #endif

                #pragma omp parallel for simd reduction(+:aux)
                for (k = 0; k < N; k++) {
                    //C[i][j] += A[i][k] * B[k][j];
                    aux += A[i*N+k] * B[k*N+j];
                }

                C[i*N+j] = aux;
            }
        }

        if(myRank == 0){
            //Verifica se a carga de trabalho não foi exata
            if(numProcs*cargaTrabalho != N){
                //Ve o quanto falta computar
                int faltantes = N - numProcs*cargaTrabalho;
                //Computa o que falta

                #pragma omp parallel for collapse(2)
                for (i = N-1; i >= N-faltantes; i--) {
                    for (j = 0; j < N; j++) {
                        int aux = 0;

                        #pragma omp parallel for simd reduction(+:aux)
                        for (k = 0; k < N; k++) {
                            aux += Aoriginal[i*N+k] * B[k*N+j];
                        }

                        Cfinal[i*N+j] = aux;
                    }
                }
            }
        }

        //O processo 0 recebe as partes de cada processo
        if(MPI_Gather(C, N*cargaTrabalho, MPI_INT, Cfinal, N*cargaTrabalho, MPI_INT, 0, USED_PROCESS) != MPI_SUCCESS){
            printf("GATHER ERROR\n");
            exit(1);
        }

        //Termina o timer
        if(myRank == 0){
            endTime = omp_get_wtime();
        }

        //Printa o resultado Final
        if(myRank == 0){
            for(i = 0; i < N; i++){
                for(j = 0; j < N; j++){
                    //printf("%d ",Cfinal[i][j]);
                    printf("%d ",Cfinal[i*N+j]);
                }
                printf("\n");
            }

            printf("Tempo passado: %lf\n", endTime - startTime);
        }
    }

    //Espera todos processos chegarem aqui antes de finalizar as execuções
    MPI_Barrier(MPI_COMM_WORLD);
    if(myRank < N){
        //free(A[0]);
        free(A);
        //free(B[0]);
        free(B);
        //free(C[0]);
        free(C);
        if(myRank == 0){
            //free(Aoriginal[0]);
            free(Aoriginal);
            //free(Cfinal[0]);
            free(Cfinal);
        }
    }
    MPI_Finalize();
    return 0;
}