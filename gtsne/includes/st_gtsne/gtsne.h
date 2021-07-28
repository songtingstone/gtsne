/*
 *  gtsne.h
 *  Header file for GTSNE.
 *
 *  Created by Songting Shi.
 *  Copyright 2021. All rights reserved.
 *
 */


#ifndef GTSNE_H
#define GTSNE_H


static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }


class GTSNE
{
public:
    void run(double* X, double* Z, double *Kmeans_centers, int N, int K, int D, int D_Z, double* Y, int no_dims, double alpha, double beta,
        double perplexity, double theta, unsigned int seed, bool verbose);
    bool load_data(double** data, int* n, int* d, double* theta, double* perplexity);
    void save_data(double* data, int* landmarks, double* costs, int n, int d);
    void symmetrizeMatrix(int** row_P, int** col_P, double** val_P, int N); // should be static?!
    void R_P(double* Z, double* C, int N, int D, int K,  double* R, double* P_macro);

private:
    void computeGradient(int* inp_row_P, int* inp_col_P, double* inp_val_P, double* P_macro,
        double* R, double* R_colsum, double alpha, double beta, double* Y, double* C, int N, int K, int D, double* dC, double theta);
    void computeExactGradient(double* P, double* P_macro, double* R, double* R_colsum, double alpha, double beta,
        double* Y, double* C, int N, int K, int D, double* dC);
    double evaluateError(double* P, double* P_macro, double* R, double* Y, double* C, int N, int K, int D, double alpha, double beta);
    double evaluateError(int* row_P, int* col_P, double* val_P, double* P_macro, double* R, double* Y, double* C,
    int N, int K, int D, double alpha, double beta, double theta);
    void computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity);
    void computeGaussianPerplexity(double* X, int N, int D, int** _row_P, int** _col_P, double** _val_P, double perplexity, int K, bool verbose);
    void computeGaussianPerplexity(double* X, int N, int D, int** _row_P, int** _col_P, double** _val_P, double perplexity, double threshold);
    void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD);
    double randn();
    void zeroMean(double* X, int N, int D);
    void check_nan(char* info, double* A, int N, int D);
    void print_matrix(char* info, double* A, int N, int D);
};

#endif
