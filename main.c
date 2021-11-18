#include<string.h>
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include <omp.h>
#include"mpi.h"

//Matrix N X N
#define N 10
//Número de Threads para paralelização
#define NUMTHREADS 4

int main(int argc, char **argv){
    double wtime;
    int i,j;
    int myRank,numProcs;
    int divisaoTrab;
    int **a,**b,**c,*aContinuo,*bContinuo,*cContinuo,**aPedaco,*aPedacoContinuo,**bPedaco,*bPedacoContinuo,**cPedaco,*cPedacoContinuo;
    MPI_Datatype pedacoMatrix;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
    MPI_Comm_size(MPI_COMM_WORLD,&numProcs);

    //Tamanho da matriz não é divisivel pelo número de processos
    /* ***********************************************************
        O QUE FAZER QUANDO TEMOS MUITOS PROCESSOS OU QUANDO TEMOS UM NÚMERO NÃO EXATO NA DIVISÃO DO TRABALHO?
    ***********************************************************  */
    if(N % numProcs != 0 || numProcs > N){
        printf("Número de processos inválido\n");
        printf("Número de processos corretos deve ser %d\n",N);
        return 0;
    }

    //Já há processos suficientes
    //if(myRank >= N)
    //    return 0;

    //Cada processo pega N/numProcs linhas e colunas
    divisaoTrab = N/numProcs;
    
    if(myRank == 0){
        wtime = omp_get_wtime();
        //Alocando as arrays para ficar continuo na memória
        aContinuo = (int *)calloc(N*N,sizeof(int));
        bContinuo = (int *)calloc(N*N,sizeof(int));
        cContinuo = (int *)calloc(N*N,sizeof(int));
        //Alocando as matrizes
        a = (int **)calloc(N,sizeof(int *));
        b = (int **)calloc(N,sizeof(int *));
        c = (int **)calloc(N,sizeof(int *));
        //Colocando os ponteiros nas matrizes para apontarem para as arrays continuas
        for(i = 0; i < N; i++) {
            a[i] = &(aContinuo[i*N]);
            b[i] = &(bContinuo[i*N]);
            c[i] = &(cContinuo[i*N]);
        }
        //Preenche as matrizes
        int k = 0;
        for(i = 0; i < N; i++) {
            for(j = 0; j < N; j++) {
                /* ***********************************************************    
                NÃO TÁ DO JEITO QUE TEM QUE SER COM SRAND(2021)
                DETALHE QUE NO a COLOQUEI POR LINHA E NO b COLOQUEI POR COLUNA
                PARA ASSIM NO a AS LINHAS FICAREM CONTINUAS NA MEMÓRIA E NO b AS COLUNAS FICAREM CONTINUAS
                ***********************************************************  */
                a[i][j] = k;
                b[j][i] = k;
                k++;
            }
        }
    }

    /************************************************************  
     * DEVO CALCULAR A CARGA DE TRABALHO NO PROCESSO 0 E ENVIAR PARA TODOS? (PARECE MEIO PERDA DE TEMPO 
     * JÁ QUE BASTA QUE CADA UM FAÇA A SIMPLES CONTA N/numProcs)
     ************************************************************/
    //Manda para todos os processos qual vai ser sua carga de trabalho
    //MPI_Bcast(&divisaoTrab, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //Aloca as arrays para ficar continua na memória
    aPedacoContinuo = (int *)calloc(N*divisaoTrab,sizeof(int));
    bPedacoContinuo = (int *)calloc(N*divisaoTrab,sizeof(int));
    cPedacoContinuo = (int *)calloc(N*N,sizeof(int));
    //Aloca a matrix
    aPedaco = (int **)calloc(divisaoTrab,sizeof(int *));
    bPedaco = (int **)calloc(divisaoTrab,sizeof(int *));
    cPedaco = (int **)calloc(N,sizeof(int *));
    //Arruma os ponteiros
    for(i= 0; i< divisaoTrab; i++){
        aPedaco[i] = &(aPedacoContinuo[i*N]);
        bPedaco[i] = &(bPedacoContinuo[i*N]);
    }
    for(int i=0;i<N;i++)
        cPedaco[i] = &(cPedacoContinuo[i*N]);

    //Manda para cada processo seu pedaço da matrix A
    MPI_Scatter(aContinuo, N, MPI_INT, &(aPedaco[0][0]), N, MPI_INT, 0, MPI_COMM_WORLD);
    //Manda para cada processo seu pedaço da matrix B
    MPI_Scatter(bContinuo, N, MPI_INT, &(bPedaco[0][0]), N, MPI_INT, 0, MPI_COMM_WORLD);


    //Faz as multiplicações (PARALELIZAR AQUI SÓ?)
    //#pragma omp parallel for num_threads(NUMTHREADS)
    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            cPedaco[i][j] = aPedaco[0][j] * bPedaco[0][i];
        }
    }


    //Reduz a matrix C de cada processo somando os resultados
    MPI_Reduce(cPedacoContinuo, cContinuo, N*N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    /************************************************************  
     * NÃO PRECISA USAR GATHER?
     * ************************************************************/
    //MPI_Gather(cContinuo,1,pedacoMatrix,&(c[divisaoTrab][divisaoTrab]),1,pedacoMatrix,0,MPI_COMM_WORLD);

    if(myRank == 0){
        for(i = 0; i < N; i++) {
            for(j = 0; j < N; j++) {
                printf("%d  ", c[i][j]);
            }
            printf("\n");
        }
    }

    /*for(i = 0; i < N; i++) {
        if(i == 0) {
            printf("rank = %d\n", myRank);
        }
        for(j = 0; j < N; j++) {
            printf("%d  ", cPedaco[i][j]);
        }
        printf("\n");
    }*/
    

    //Antes de limpar as coisas espera todos terminarem
    MPI_Barrier(MPI_COMM_WORLD);;
    free(aPedaco);
    free(aPedacoContinuo);
    free(bPedaco);
    free(bPedacoContinuo);
    free(cPedaco);
    free(cPedacoContinuo);
    if(myRank==0){
        wtime = omp_get_wtime() - wtime;
        printf("\n\nTEMPO: %f\n",wtime);
        free(aContinuo);
        free(a);
        free(bContinuo);
        free(b);
        free(cContinuo);
        free(c);
    }
    return 0;

}