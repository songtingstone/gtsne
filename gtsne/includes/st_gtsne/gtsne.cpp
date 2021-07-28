/*
 *  gtsne.cpp
 *  Implementation of both standard and Barnes-Hut-GTSNE.
 *
 *  Created by Songting Shi.
 *  Copyright 2021. All rights reserved.
 *
 */

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <time.h>
#include "quadtree.h"
#include "vptree.h"
#include "gtsne.h"

extern "C" {
    #include <cblas.h>
}


using namespace std;

// Perform t-SNE
void GTSNE::run(double* X, double* Z, double *Kmeans_centers, int N, int K, int D, int D_Z, double* Y, int no_dims,
    double alpha, double beta,
    double perplexity, double theta, unsigned int seed, bool verbose) {
    // Initalize the pseudorandom number generator
    srand(seed);

    // Determine whether we are using an exact algorithm
    if(N - 1 < 3 * perplexity) { printf("Perplexity too large for the number of data points!\n"); exit(1); }

    if (verbose) {
        printf("Using no_dims = %d, perplexity = %f, theta = %f, seed=%d\n", no_dims, perplexity, theta, seed);
    }

    bool exact = (theta == .0) ? true : false;

    // scaling beta
    beta = beta/(1.*N);
    // Set learning parameters
    float total_time = .0;
    clock_t start, end;
	int max_iter = 1000, stop_lying_iter = 250, mom_switch_iter = 250;
	double momentum = .5, final_momentum = .8;
	double eta = 200.0;

    // Allocate some memory
    double* dY    = (double*) malloc(N * no_dims * sizeof(double));
    double* uY    = (double*) malloc(N * no_dims * sizeof(double));
    double* gains = (double*) malloc(N * no_dims * sizeof(double));
    if(dY == NULL || uY == NULL || gains == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int i = 0; i < N * no_dims; i++)    uY[i] =  .0;
    for(int i = 0; i < N * no_dims; i++) gains[i] = 1.0;

    // Normalize input data (to prevent numerical problems)
    if (verbose) {
        printf("Computing input similarities...\n");
    }

    start = clock();
    zeroMean(X, N, D);
    double max_X = .0;
    for(int i = 0; i < N * D; i++) {
        if(X[i] > max_X) max_X = X[i];
    }

    for(int i = 0; i < N * D; i++) X[i] /= max_X;

    // Compute input similarities for exact GTSNE
    double* P; double* P_macro; int* row_P; int* col_P; double* val_P;
    double* C; double* R; double* R_colsum;

    P_macro = (double*) malloc(K * K * sizeof(double));
    C = (double*) malloc(K * no_dims * sizeof(double));
    R = (double*) malloc(N * K * sizeof(double));
    R_colsum = (double*) malloc(K * sizeof(double));
    if(P_macro == NULL || C == NULL || R == NULL || R_colsum == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    R_P(Z, Kmeans_centers, N, D_Z, K,  R, P_macro);
    // compute R_colsum
    for(int k=0; k<K; k++) R_colsum[k] = .0;
    for(int i=0; i<K*no_dims; i++) C[i] = .0;
    for(int i=0; i<N; i++){
        int ind_iK = i*K;
        for(int k=0; k<K; k++){
            R_colsum[k] += R[ind_iK + k];
        }
    }
    if(exact) {
        // Compute similarities
        P = (double*) malloc(N * N * sizeof(double));
        if( P == NULL ) { printf("Memory allocation failed!\n"); exit(1); }
        computeGaussianPerplexity(X, N, D, P, perplexity);
        // Symmetrize input similarities
        if (verbose) {
            printf("Symmetrizing...\n");
        }
        for(int n = 0; n < N; n++) {
            for(int m = n + 1; m < N; m++) {
                P[n * N + m] += P[m * N + n];
                P[m * N + n]  = P[n * N + m];
            }
        }
        double sum_P = .0;
        for(int i = 0; i < N * N; i++) sum_P += P[i];
        for(int i = 0; i < N * N; i++) P[i] /= sum_P;
    }

    // Compute input similarities for approximate GTSNE
    else {
        // Compute asymmetric pairwise input similarities
        computeGaussianPerplexity(X, N, D, &row_P, &col_P, &val_P, perplexity, (int) (3 * perplexity), verbose);

        // Symmetrize input similarities
        symmetrizeMatrix(&row_P, &col_P, &val_P, N);
        double sum_P = .0;
        for(int i = 0; i < row_P[N]; i++) sum_P += val_P[i];
        for(int i = 0; i < row_P[N]; i++) val_P[i] /= sum_P;
    }
    end = clock();

    // Lie about the P-values
    if(exact) { for(int i = 0; i < N * N; i++)        P[i] *= 12.0; }
    else {      for(int i = 0; i < row_P[N]; i++) val_P[i] *= 12.0; }

	// Initialize solution (randomly)
	for(int i = 0; i < N * no_dims; i++) Y[i] = randn() * .0001;

	// Perform main training loop
    if (verbose) {
        if(exact) printf("Done in %4.2f seconds!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC);
        else printf("Done in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC, (double) row_P[N] / ((double) N * (double) N));
    }
    start = clock();
	for(int iter = 0; iter < max_iter; iter++) {

        // Compute (approximate) gradient
//        check_nan("run before compute exact gradient",Y,N,no_dims);
        if(exact) computeExactGradient(P, P_macro, R, R_colsum, alpha, beta,
            Y, C, N, K, no_dims, dY);
        else computeGradient(row_P, col_P, val_P, P_macro, R, R_colsum, alpha, beta, Y, C, N, K, no_dims, dY, theta);

//        check_nan("dY",dY,N,no_dims);
        // Update gains
        for(int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
        for(int i = 0; i < N * no_dims; i++) if(gains[i] < .01) gains[i] = .01;

        // Perform gradient update (with momentum and gains)
        for(int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
		for(int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];

        // Make solution zero-mean
		zeroMean(Y, N, no_dims);

        // Stop lying about the P-values after a while, and switch momentum
        if(iter == stop_lying_iter) {
            if(exact) { for(int i = 0; i < N * N; i++)        P[i] /= 12.0; }
            else      { for(int i = 0; i < row_P[N]; i++) val_P[i] /= 12.0; }
        }
        if(iter == mom_switch_iter) momentum = final_momentum;

        // Print out progress
        if(verbose){
             if(( ( (iter > 0) && ((iter%50) == 0) ) || (iter == (max_iter - 1)) )) {
                end = clock();
                double Cost = .0;
                if(exact) Cost = evaluateError(P, P_macro, R, Y, C, N, K, no_dims, alpha, beta);
                else      Cost = evaluateError(row_P, col_P, val_P, P_macro, R, Y, C, N, K, no_dims, alpha, beta, theta);  // doing approximate computation here!
                if(iter == 0)
                    printf("Iteration %d: error is %f\n", iter + 1, Cost);
                else {
                    total_time += (float) (end - start) / CLOCKS_PER_SEC;
                    printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, Cost, (float) (end - start) / CLOCKS_PER_SEC);
                }
                start = clock();
            }
        }
    }
    end = clock(); total_time += (float) (end - start) / CLOCKS_PER_SEC;

    // Clean up memory
    free(dY);
    free(uY);
    free(gains);
    free(P_macro);
    free(C);
    free(R);
    free(R_colsum);
    if(exact) free(P);
    else {
        free(row_P); row_P = NULL;
        free(col_P); col_P = NULL;
        free(val_P); val_P = NULL;
    }
    if (verbose) {
        printf("Fitting performed in %4.2f seconds.\n", total_time);
    }
}


// Compute gradient of the GTSNE cost function (using Barnes-Hut algorithm)
void GTSNE::computeExactGradient(double* P, double* P_macro, double* R, double* R_colsum, double alpha, double beta,
    double* Y, double* C, int N, int K, int D, double* dC){
    // Make sure the current gradient contains zeros
	for(int i = 0; i < N * D; i++) dC[i] = 0.0;

    // compute C
    // C = R^TY
    // Y DxN; R: KxN C:DxK
    // C = YR^T/R_colsum
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, D, K, N, 1.0, Y, D, R, K, 0.0, C, D);
    for(int k=0; k<K; k++){
        int ind_kD = k*D;
        for(int d=0; d<D; d++){
           C[ind_kD + d] /= R_colsum[k];
        }
    }

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    double* DDC = (double*) malloc(K * K * sizeof(double));
    if(DD == NULL || DDC == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(Y, N, D, DD);
    computeSquaredEuclideanDistance(C, K, D, DDC);
//    print_matrix("DDC",DDC, K, K);
//    check_nan("DDC", DDC, K, K);
    // Compute Q-matrix and normalization sum
    double* Q    = (double*) malloc(N * N * sizeof(double));
    double* Q_macro    = (double*) malloc(K * K * sizeof(double));
    if(Q == NULL || Q_macro == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    double sum_Q = .0, sum_Q_macro = .0;
    for(int n = 0; n < N; n++) {
        int ind_nN = n*N;
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[ ind_nN + m ] = 1 / (1 + DD[ ind_nN  + m ]);
                sum_Q += Q[ ind_nN + m];
            }else{
                Q[ ind_nN + m] = 0;
            }
        }
    }

    for(int k = 0; k < K; k++) {
        int ind_kK = k*K;
    	for(int l = 0; l < K; l++) {
            if(k != l) {
                Q_macro[ind_kK + l] = 1. / (1. + DDC[ind_kK + l]);
                sum_Q_macro += Q_macro[ind_kK + l];
            }else{
                Q_macro[ind_kK + l] = 0;
            }

        }
    }
	// Perform the computation of the gradient
//	dC = 2(p_ij-q_ij)Q_ij(y_i - y_j) + alpha (p_macro_kl - q_macro_kl) Q_macro_kl (R_ik - Ril)(C_k - C_l) + beta R_ik (y_i-c_k)
	for(int n = 0; n < N; n++) {
	    int ind_nN = n*N;
	    int ind_nD = n*D;
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                double mult = (P[ind_nN + m] - (Q[ind_nN + m] / sum_Q)) * Q[ind_nN + m];
                for(int d = 0; d < D; d++) {
                    dC[ind_nD + d] += (Y[ind_nD + d] - Y[m * D + d]) * mult;
                }
            }
		}
	}
	// for macro part
	for(int i = 0; i < N; i++){
	    int ind_iD = i*D;
	    int ind_iK = i*K;
        for(int k = 0; k < K; k++) {
	        int ind_kK = k*K;
	        int ind_kD = k*D;
    	    for(int l = 0; l < K; l++) {
    	        int ind_lD = l*D;
                if(k != l) {
                    double mult = alpha * (P_macro[ind_kK + l] - (Q_macro[ind_kK + l] / sum_Q_macro)) * Q_macro[ind_kK + l] * (R[ind_iK +k] - R[ind_iK +l]) ;
                    for(int d = 0; d < D; d++) {
                        dC[ind_iD + d] += (C[ind_kD + d] - C[ind_lD + d] ) * mult;
                    }
                }
		    }
	    }
	}
	// for the k-Means part
	for(int i = 0; i < N; i++){
	     int ind_iK = i*K;
	     int ind_iD = i*D;
         for(int k = 0; k < K; k++) {
            int ind_kD = k*D;
            double mult = beta * R[ind_iK +k];
            for(int d = 0; d < D; d++) {
                dC[ind_iD + d]  += (Y[ ind_iD + d] - C[ind_kD + d]) * mult;
            }
         }
	}

    // Free memory
    free(DD); DD = NULL;
    free(DDC); DDC = NULL;
    free(Q);  Q  = NULL;
    free(Q_macro);  Q_macro  = NULL;
}

void GTSNE::computeGradient(int* inp_row_P, int* inp_col_P, double* inp_val_P, double* P_macro,
  double* R, double* R_colsum, double alpha, double beta, double* Y, double* C, int N, int K, int D, double* dC, double theta)
{
    // Y --> C
    // Y DxN; R: KxN C:DxK
    // C = YR^T/R_colsum
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, D, K, N, 1.0, Y, D, R, K, 0.0, C, D);
    for(int k=0; k<K; k++){
        int ind_kD = k*D;
        for(int d=0; d<D; d++){
           C[ind_kD + d] /= R_colsum[k];
        }
    }

    double* DDC = (double*) malloc(K * K * sizeof(double));
    if( DDC == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(C, K, D, DDC);
//    check_nan("DDC", DDC, K, K);
    // Compute Q-matrix and normalization sum
    double* Q_macro    = (double*) malloc(K * K * sizeof(double));
    if(Q_macro == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    double sum_Q_macro = .0;
    for(int k = 0; k < K; k++) {
        int ind_kK = k*K;
    	for(int l = 0; l < K; l++) {
            if(k != l) {
                Q_macro[ind_kK + l] = 1. / (1. + DDC[ind_kK + l]);
                sum_Q_macro += Q_macro[ind_kK + l];
            }else{
                Q_macro[ind_kK + l] = 0;
            }

        }
    }

    // Construct quadtree on current map
    QuadTree* tree = new QuadTree(Y, N);

    // Compute all terms required for GTSNE gradient
    double sum_Q = .0;
    double* pos_f = (double*) calloc(N * D, sizeof(double));
    double* neg_f = (double*) calloc(N * D, sizeof(double));
    if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);
    for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q);

    // Compute final GTSNE gradient
    for(int i = 0; i < N * D; i++) {
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
    }

   // for macro part
	for(int i = 0; i < N; i++){
	    int ind_iD = i*D;
	    int ind_iK = i*K;
        for(int k = 0; k < K; k++) {
	        int ind_kK = k*K;
	        int ind_kD = k*D;
    	    for(int l = 0; l < K; l++) {
    	        int ind_lD = l*D;
                if(k != l) {
                    double mult = alpha * (P_macro[ind_kK + l] - (Q_macro[ind_kK + l] / sum_Q_macro)) * Q_macro[ind_kK + l] * (R[ind_iK +k] - R[ind_iK +l]) ;
                    for(int d = 0; d < D; d++) {
                        dC[ind_iD + d] += (C[ind_kD + d] - C[ind_lD + d] ) * mult;
                    }
                }
		    }
	    }
	}
	// for the k-Means part
	for(int i = 0; i < N; i++){
	     int ind_iK = i*K;
	     int ind_iD = i*D;
         for(int k = 0; k < K; k++) {
            int ind_kD = k*D;
            double mult = beta * R[ind_iK +k];
            for(int d = 0; d < D; d++) {
                dC[ind_iD + d]  += (Y[ ind_iD + d] - C[ind_kD + d]) * mult;
            }
         }
	}

    // Free memory
    free(DDC); DDC = NULL;
    free(Q_macro);  Q_macro  = NULL;
    free(pos_f);
    free(neg_f);
    delete tree;
}



// Evaluate GTSNE cost function (exactly)
double GTSNE::evaluateError(double* P, double* P_macro, double* R, double* Y, double* C, int N, int K, int D, double alpha, double beta) {

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    double* DDC = (double*) malloc(K * K * sizeof(double));
    double* Q = (double*) malloc(N * N * sizeof(double));
    double* Q_macro = (double*) malloc(K * K * sizeof(double));
    double* tmp_y = (double*) malloc( 2 * sizeof(double) );
    if(DD == NULL || DDC == NULL || Q == NULL || Q_macro == NULL || tmp_y == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(Y, N, 2, DD); // Y colmajor 2xN <==> rowmajor N x 2 ?
    computeSquaredEuclideanDistance(C, K, 2, DDC);

    // cost = P log P/Q + alpha P_macro log P_macro / Q_macro + beta R log || Z - C||^2
    // Compute Q-matrix and normalization sum
    double sum_Q = DBL_MIN;
    for(int n = 0; n < N; n++) {
        int ind_nN = n*N;
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[ind_nN + m] = 1 / (1 + DD[ind_nN + m]);
                sum_Q += Q[ind_nN + m];
            }
            else Q[ind_nN + m] = DBL_MIN;
        }
    }
    for(int i = 0; i < N * N; i++) Q[i] /= sum_Q;

    double sum_Q_macro = DBL_MIN;
    for(int k = 0; k < K; k++) {
        int ind_kK = k*K;
    	for(int l = 0; l < K; l++) {
            if(k != l) {
                Q_macro[ind_kK + l] = 1 / (1 + DDC[ind_kK + l]);
                sum_Q_macro += Q_macro[ind_kK + l];
            }
            else Q_macro[ind_kK + l] = DBL_MIN;
        }
    }
    for(int i = 0; i < K * K; i++) Q_macro[i] /= sum_Q_macro;


    // Sum GTSNE error
    double Cost = .0;
	for(int n = 0; n < N; n++) {
	    int ind_nN = n*N;
		for(int m = 0; m < N; m++) {
			Cost += P[ind_nN + m] * log((P[ind_nN + m] + 1e-9) / (Q[ind_nN + m] + 1e-9));
		}
	}

    double C_macro = .0;
	for(int k = 0; k < K; k++) {
	    int ind_kK = k * K;
		for(int l = 0; l < K; l++) {
			C_macro += P_macro[ ind_kK + l ] * log((P_macro[ ind_kK + l] + 1e-9) / (Q_macro[ind_kK + l] + 1e-9));
		}
	}
	Cost += alpha * C_macro;

	// for k-Means loss
	double C_kMeans = .0;
	for(int i = 0; i < N; i++){
	    int ind_iD = i*D;
	    int ind_iK = i*K;
        for(int k = 0; k < K; k++) {
            int ind_kD = k*D;
            for(int d=0; d < D; d++){
                tmp_y[d] = Y[ ind_iD + d] - C[ind_kD + d];
            }
			C_kMeans += R[ind_iK + k] * cblas_ddot(D, tmp_y, 1, tmp_y, 1);
		}
	}
	Cost += beta * C_kMeans;

    // Clean up memory
    free(DD);
    free(DDC);
    free(Q);
    free(Q_macro);
    free(tmp_y);
	return Cost;
}

// Evaluate GTSNE cost function (approximately)
double GTSNE::evaluateError(int* row_P, int* col_P, double* val_P, double* P_macro, double* R, double* Y, double* C,
    int N, int K, int D, double alpha, double beta, double theta)
{
    double* DDC = (double*) malloc(K * K * sizeof(double));
    double* Q_macro = (double*) malloc(K * K * sizeof(double));
    double* tmp_y = (double*) malloc( 2 * sizeof(double) );
    if(DDC == NULL || Q_macro == NULL || tmp_y == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(C, K, 2, DDC);

    check_nan("DDC", DDC, K, K);
    // cost = P log P/Q + alpha P_macro log P_macro / Q_macro + beta R log || Z - C||^2
    // Compute Q-matrix and normalization sum

    double sum_Q_macro = DBL_MIN;
    for(int k = 0; k < K; k++) {
        int ind_kK = k*K;
    	for(int l = 0; l < K; l++) {
            if(k != l) {
                Q_macro[ind_kK + l] = 1. / (1. + DDC[ind_kK + l]);
                sum_Q_macro += Q_macro[ind_kK + l];
            }
            else Q_macro[ind_kK + l] = DBL_MIN;
        }
    }
    for(int i = 0; i < K * K; i++) Q_macro[i] /= sum_Q_macro;


    // Get estimate of normalization term
    const int QT_NO_DIMS = 2;
    QuadTree* tree = new QuadTree(Y, N);
    double buff[QT_NO_DIMS] = {.0, .0};
    double sum_Q = .0;
    for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q);

    // Loop over all edges to compute GTSNE error
    int ind1, ind2;
    double Cost = .0, Q;
    for(int n = 0; n < N; n++) {
        ind1 = n * QT_NO_DIMS;
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {
            Q = .0;
            ind2 = col_P[i] * QT_NO_DIMS;
            for(int d = 0; d < QT_NO_DIMS; d++) buff[d]  = Y[ind1 + d];
            for(int d = 0; d < QT_NO_DIMS; d++) buff[d] -= Y[ind2 + d];
            for(int d = 0; d < QT_NO_DIMS; d++) Q += buff[d] * buff[d];
            Q = (1.0 / (1.0 + Q)) / sum_Q;
            Cost += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
        }
    }

    double C_macro = .0;
	for(int k = 0; k < K; k++) {
	    int ind_kK = k * K;
		for(int l = 0; l < K; l++) {
			C_macro += P_macro[ ind_kK + l ] * log((P_macro[ ind_kK + l] + 1e-9) / (Q_macro[ind_kK + l] + 1e-9));
		}
	}
	Cost += alpha * C_macro;

	// for k-Means loss
	double C_kMeans = .0;
	for(int i = 0; i < N; i++){
	    int ind_iD = i*D;
	    int ind_iK = i*K;
        for(int k = 0; k < K; k++) {
            int ind_kD = k*D;
            for(int d=0; d < D; d++){
                tmp_y[d] = Y[ ind_iD + d] - C[ind_kD + d];
            }
			C_kMeans += R[ind_iK + k] * cblas_ddot(D, tmp_y, 1, tmp_y, 1);
		}
	}
	Cost += beta * C_kMeans;

    // Clean up memory
    free(DDC);
    free(Q_macro);
    free(tmp_y);
    delete tree;
    return Cost;
}


// Compute input similarities with a fixed perplexity
void GTSNE::computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity) {

	// Compute the squared Euclidean distance matrix
	double* DD = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeSquaredEuclideanDistance(X, N, D, DD);

	// Compute the Gaussian kernel row by row
	for(int n = 0; n < N; n++) {

		// Initialize some variables
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta =  DBL_MAX;
		double tol = 1e-5;
        double sum_P;

		// Iterate until we found a good perplexity
		int iter = 0;
		while(!found && iter < 200) {

			// Compute Gaussian kernel row
			for(int m = 0; m < N; m++) P[n * N + m] = exp(-beta * DD[n * N + m]);
			P[n * N + n] = DBL_MIN;

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for(int m = 0; m < N; m++) sum_P += P[n * N + m];
			double H = 0.0;
			for(int m = 0; m < N; m++) H += beta * (DD[n * N + m] * P[n * N + m]);
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if(Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}

		// Row normalize P
		for(int m = 0; m < N; m++) P[n * N + m] /= sum_P;
	}

	// Clean up memory
	free(DD); DD = NULL;
}


// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
void GTSNE::computeGaussianPerplexity(double* X, int N, int D, int** _row_P, int** _col_P, double** _val_P, double perplexity, int K, bool verbose) {

    if(perplexity > K) printf("Perplexity should be lower than K!\n");

    // Allocate the memory we need
    *_row_P = (int*)    malloc((N + 1) * sizeof(int));
    *_col_P = (int*)    calloc(N * K, sizeof(int));
    *_val_P = (double*) calloc(N * K, sizeof(double));
    if(*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    int* row_P = *_row_P;
    int* col_P = *_col_P;
    double* val_P = *_val_P;
    double* cur_P = (double*) malloc((N - 1) * sizeof(double));
    if(cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    row_P[0] = 0;
    for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + K;

    // Build ball tree on data set
    VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
    vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
    for(int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
    tree->create(obj_X);

    // Loop over all points to find nearest neighbors
    if (verbose) {
        printf("Building tree...\n");
    }
    vector<DataPoint> indices;
    vector<double> distances;
    for(int n = 0; n < N; n++) {

        if (verbose) {
            if(n % 10000 == 0) printf(" - point %d of %d\n", n, N);
        }

        // Find nearest neighbors
        indices.clear();
        distances.clear();
        tree->search(obj_X[n], K + 1, &indices, &distances);

        // Initialize some variables for binary search
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta =  DBL_MAX;
		double tol = 1e-5;

		// Iterate until we found a good perplexity
		int iter = 0; double sum_P;
		while(!found && iter < 200) {

			// Compute Gaussian kernel row
			for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1]);

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for(int m = 0; m < K; m++) sum_P += cur_P[m];
			double H = .0;
			for(int m = 0; m < K; m++) H += beta * (distances[m + 1] * cur_P[m]);
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if(Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}

		// Row-normalize current row of P and store in matrix
        for(int m = 0; m < K; m++) cur_P[m] /= sum_P;
        for(int m = 0; m < K; m++) {
            col_P[row_P[n] + m] = indices[m + 1].index();
            val_P[row_P[n] + m] = cur_P[m];
        }
    }

    // Clean up memory
    obj_X.clear();
    free(cur_P);
    delete tree;
}


// Compute input similarities with a fixed perplexity (this function allocates memory another function should free)
void GTSNE::computeGaussianPerplexity(double* X, int N, int D, int** _row_P, int** _col_P, double** _val_P, double perplexity, double threshold) {

    // Allocate some memory we need for computations
    double* buff  = (double*) malloc(D * sizeof(double));
    double* DD    = (double*) malloc(N * sizeof(double));
    double* cur_P = (double*) malloc(N * sizeof(double));
    if(buff == NULL || DD == NULL || cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

    // Compute the Gaussian kernel row by row (to find number of elements in sparse P)
    int total_count = 0;
	for(int n = 0; n < N; n++) {

        // Compute the squared Euclidean distance matrix
        for(int m = 0; m < N; m++) {
            for(int d = 0; d < D; d++) buff[d]  = X[n * D + d];
            for(int d = 0; d < D; d++) buff[d] -= X[m * D + d];
            DD[m] = .0;
            for(int d = 0; d < D; d++) DD[m] += buff[d] * buff[d];
        }

		// Initialize some variables
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta =  DBL_MAX;
		double tol = 1e-5;

		// Iterate until we found a good perplexity
		int iter = 0; double sum_P;
		while(!found && iter < 200) {

			// Compute Gaussian kernel row
			for(int m = 0; m < N; m++) cur_P[m] = exp(-beta * DD[m]);
			cur_P[n] = DBL_MIN;

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for(int m = 0; m < N; m++) sum_P += cur_P[m];
			double H = 0.0;
			for(int m = 0; m < N; m++) H += beta * (DD[m] * cur_P[m]);
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if(Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}

		// Row-normalize and threshold current row of P
        for(int m = 0; m < N; m++) cur_P[m] /= sum_P;
        for(int m = 0; m < N; m++) {
            if(cur_P[m] > threshold / (double) N) total_count++;
        }
    }

    // Allocate the memory we need
    *_row_P = (int*)    malloc((N + 1)     * sizeof(int));
    *_col_P = (int*)    malloc(total_count * sizeof(int));
    *_val_P = (double*) malloc(total_count * sizeof(double));
    int* row_P = *_row_P;
    int* col_P = *_col_P;
    double* val_P = *_val_P;
    if(row_P == NULL || col_P == NULL || val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    row_P[0] = 0;

    // Compute the Gaussian kernel row by row (this time, store the results)
    int count = 0;
	for(int n = 0; n < N; n++) {

        // Compute the squared Euclidean distance matrix
        for(int m = 0; m < N; m++) {
            for(int d = 0; d < D; d++) buff[d]  = X[n * D + d];
            for(int d = 0; d < D; d++) buff[d] -= X[m * D + d];
            DD[m] = .0;
            for(int d = 0; d < D; d++) DD[m] += buff[d] * buff[d];
        }

		// Initialize some variables
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta =  DBL_MAX;
		double tol = 1e-5;

		// Iterate until we found a good perplexity
		int iter = 0; double sum_P;
		while(!found && iter < 200) {

			// Compute Gaussian kernel row
			for(int m = 0; m < N; m++) cur_P[m] = exp(-beta * DD[m]);
			cur_P[n] = DBL_MIN;

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for(int m = 0; m < N; m++) sum_P += cur_P[m];
			double H = 0.0;
			for(int m = 0; m < N; m++) H += beta * (DD[m] * cur_P[m]);
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if(Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}

		// Row-normalize and threshold current row of P
		for(int m = 0; m < N; m++) cur_P[m] /= sum_P;
        for(int m = 0; m < N; m++) {
            if(cur_P[m] > threshold / (double) N) {
                col_P[count] = m;
                val_P[count] = cur_P[m];
                count++;
            }
        }
        row_P[n + 1] = count;
	}

    // Clean up memory
    free(DD);    DD = NULL;
    free(buff);  buff = NULL;
    free(cur_P); cur_P = NULL;
}


void GTSNE::symmetrizeMatrix(int** _row_P, int** _col_P, double** _val_P, int N) {

    // Get sparse matrix
    int* row_P = *_row_P;
    int* col_P = *_col_P;
    double* val_P = *_val_P;

    // Count number of elements and row counts of symmetric matrix
    int* row_counts = (int*) calloc(N, sizeof(int));
    if(row_counts == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) present = true;
            }
            if(present) row_counts[n]++;
            else {
                row_counts[n]++;
                row_counts[col_P[i]]++;
            }
        }
    }
    int no_elem = 0;
    for(int n = 0; n < N; n++) no_elem += row_counts[n];

    // Allocate memory for symmetrized matrix
    int*    sym_row_P = (int*)    malloc((N + 1) * sizeof(int));
    int*    sym_col_P = (int*)    malloc(no_elem * sizeof(int));
    double* sym_val_P = (double*) malloc(no_elem * sizeof(double));
    if(sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for(int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + row_counts[n];

    // Fill the result matrix
    int* offset = (int*) calloc(N, sizeof(int));
    if(offset == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) {
                    present = true;
                    if(n <= col_P[i]) {                                                 // make sure we do not add elements twice
                        sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                        sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
                        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
                    }
                }
            }

            // If (col_P[i], n) is not present, there is no addition involved
            if(!present) {
                sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
                sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
            }

            // Update offsets
            if(!present || (present && n <= col_P[i])) {
                offset[n]++;
                if(col_P[i] != n) offset[col_P[i]]++;
            }
        }
    }

    // Divide the result by two
    for(int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

    // Return symmetrized matrices
    free(*_row_P); *_row_P = sym_row_P;
    free(*_col_P); *_col_P = sym_col_P;
    free(*_val_P); *_val_P = sym_val_P;

    // Free up some memery
    free(offset); offset = NULL;
    free(row_counts); row_counts  = NULL;
}

// Compute squared Euclidean distance matrix (using BLAS)
void GTSNE::computeSquaredEuclideanDistance(double* X, int N, int D, double* DD) {
    double* dataSums = (double*) calloc(N, sizeof(double));
    if(dataSums == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            dataSums[n] += (X[n * D + d] * X[n * D + d]);
        }
    }
    for(int n = 0; n < N; n++) {
        for(int m = 0; m < N; m++) {
            DD[n * N + m] = dataSums[n] + dataSums[m];
        }
    }
    // X is DxN colmajor
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, N, N, D, -2.0, X, D, X, D, 1.0, DD, N);
    free(dataSums); dataSums = NULL;
}



// Makes data zero-mean
void GTSNE::zeroMean(double* X, int N, int D) {

	// Compute data mean
	double* mean = (double*) calloc(D, sizeof(double));
    if(mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			mean[d] += X[n * D + d];
		}
	}
	for(int d = 0; d < D; d++) {
		mean[d] /= (double) N;
	}

	// Subtract data mean
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			X[n * D + d] -= mean[d];
		}
	}
    free(mean); mean = NULL;
}


void GTSNE::R_P(double* Z, double* C, int N, int D, int K,  double* R, double* P_macro){
    // compute the R, and P_macro
    // R_{ki} =  1/( || c_k - z_i ||^2 + 1 )
    // \tilde P_macro(k,l) = 1/( || c_k - c_l ||^2 + 1 )
    // P_macro(k,l) = \tilde P_macro(k,l)/Z; Z = sum_{k,l} \tilde P_macro(k,l)
    int iter_max = 1000;
    double* Z_sum = (double*)  malloc(N * sizeof(double));
    double* C_sum = (double*)  malloc(K * sizeof(double));
    double* R_rowsum = (double*)  malloc(N * sizeof(double));
    double* R_colsum = (double*)  malloc(K * sizeof(double));
    double* ZC = (double*)  malloc( N * K * sizeof(double));
    double* DistC = (double*)  malloc( K * K * sizeof(double));
    if(Z_sum == NULL || C_sum == NULL || R_rowsum == NULL || R_colsum == NULL || ZC == NULL || DistC == NULL)
        { printf("Memory allocated failed! exit"); exit(1); }
    for(int i=0; i < N; i++ ) { Z_sum[i]=0; R_rowsum[i]=0; }
    for(int k=0; k < K; k++ ) { R_colsum[k]=0; }

    for(int i=0; i < N; i++){
        Z_sum[i] = cblas_ddot(D, Z+i*D, 1, Z+i*D, 1);
    }

    for(int k=0; k < K; k++){
        C_sum[k] = cblas_ddot(D, C+k*D, 1, C+k*D, 1);
    }
    for(int i=0; i<N; i++){
        int ind_i = i*K;
        for(int k=0; k<K; k++){
            R[ ind_i + k ] = Z_sum[i] + C_sum[k];
        }
    }
    // R_ki = Z_i^2 + C_k^2 - 2 Z_i C_k
    // R: KxN; Z: DxN; C: DxK
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, K, N, D, -2., C, D, Z, D, 1.0, R, K);

//    print_matrix("C ", C, K, D);
//    print_matrix("dist R", R, 3,K);
    double scale = 4./(D*D*1.);
    for(int i=0; i<N*K; i++)  R[i] = 1./(1. + scale* R[i]);
    int ind_i;
    for(int i=0; i<N; i++){
        R_rowsum[i] = .0;
        ind_i = i*K;
        for(int k=0; k < K; k++ ){
            R_rowsum[i] +=  R[ind_i + k];
        }
    }
    for(int i=0; i<N; i++){
        ind_i = i*K;
        for(int k=0; k < K; k++ ){
            R[ind_i + k] /= R_rowsum[i] + 1e-9;
        }
    }

    // sparse
    double epsilon = 1e-6;
    for(int i=0; i< N*K; i++){
        if(R[i] < epsilon) R[i]=0;
    }

    // calculate P_macro
    // P_macro K x K
    for(int k=0; k < K; k++){
        C_sum[k] = cblas_ddot(D, C+k*D, 1, C+k*D, 1);
    }
    for(int k1=0; k1<K; k1++){
        int ind_k1K = k1*K;
        for(int k2=0; k2<K; k2++){
            DistC[ind_k1K + k2] = C_sum[k1] + C_sum[k2];
        }
    }
    // C: DxK
    // C^TC
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, K, K, D, -2., C, D, C, D, 1.0, DistC, K);
    for(int k1=0; k1<K; k1++){
        int ind_k1K = k1*K;
        for(int k2=0; k2<K; k2++){
            if(k1!=k2){
//                P_macro[ind_k1K + k2] = 1./(1. + scale*DistC[ind_k1K + k2]);
                P_macro[ind_k1K + k2] = 1./(1. + DistC[ind_k1K + k2]);
            }else{
                P_macro[ind_k1K + k2] = DBL_MIN;
            }

        }
    }
    double P_sum = DBL_MIN;
    for(int i=0; i<K*K; i++) P_sum += P_macro[i];
    for(int i=0; i<K*K; i++) P_macro[i] /= P_sum;

    // sparse P_macro
    epsilon = 1e-9;
    for(int i=0; i< K*K; i++){
        if(P_macro[i] < epsilon) P_macro[i]=0;
    }


    free(Z_sum);
    free(C_sum);
    free(R_rowsum);
    free(R_colsum);
    free(ZC);
    free(DistC);
}


// Generates a Gaussian random number
double GTSNE::randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}

void GTSNE::check_nan(char* info, double* A, int N, int D){
    for(int i=0; i<N*D; i++){
        if(isnan(A[i])) {
            printf(info);
            printf(" ,the matrix contains nan at position %d",i);
            exit(1);
        }
    }
}

void GTSNE::print_matrix(char* info, double* A, int N, int D){
    printf(info);printf(":\n");
    for(int i=0; i<N; i++){
        int ind_iD = i * D;
        for(int d=0; d<D-1; d++){
            printf("%f\t", A[ind_iD +d]);
        }
        printf("%f\n", A[ind_iD +D-1]);
    }
}
