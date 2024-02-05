#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cvode/cvode.h>             // prototypes for CVODE fcts., consts.
#include <nvector/nvector_serial.h>  // serial N_Vector types, fcts., macros
#include <sunmatrix/sunmatrix_dense.h> // access to dense SUNMatrix
#include <sunlinsol/sunlinsol_dense.h> // access to dense SUNLinearSolver
#include <cvode/cvode_direct.h>        // access to CVDls interface

// Define constants
#define Ith(v,i)    NV_Ith_S(v,i-1)       // Ith numbers components 1..NEQ

// Define the repressilator parameters struct
typedef struct {
    double deg_m, deg_p, alpha, alpha_0, beta, n;
} repressilator_params;

// Function to compute the right-hand sides of the ODE system
static int repressilator_func(realtype t, N_Vector y, N_Vector ydot, void *user_data) {
    repressilator_params *p = (repressilator_params *)user_data;

    Ith(ydot,1) = -p->deg_m * Ith(y,1) + p->alpha / (1 + pow(Ith(y,6), p->n)) + p->alpha_0;
    Ith(ydot,2) = -p->deg_m * Ith(y,2) + p->alpha / (1 + pow(Ith(y,4), p->n)) + p->alpha_0;
    Ith(ydot,3) = -p->deg_m * Ith(y,3) + p->alpha / (1 + pow(Ith(y,5), p->n)) + p->alpha_0;
    Ith(ydot,4) = -p->deg_p * Ith(y,4) + p->beta * Ith(y,1);
    Ith(ydot,5) = -p->deg_p * Ith(y,5) + p->beta * Ith(y,2);
    Ith(ydot,6) = -p->deg_p * Ith(y,6) + p->beta * Ith(y,3);

    return 0;
}

// Function to simulate the repressilator
double* simulate_repressilator(double *params_array, double *initial_conditions, double *time_span, int num_points) {
    // Allocate memory for results
    double *results = (double *)malloc(num_points * 7 * sizeof(double));
    if (!results) {
        fprintf(stderr, "Failed to allocate memory for results\n");
        return NULL;
    }

    // Create and initialize parameters
    repressilator_params params = {params_array[0], params_array[1], params_array[2], params_array[3], params_array[4], params_array[5]};
    
    // Initialize N_Vector for state variables
    N_Vector y = N_VNew_Serial(6);
    if (y == NULL) { /* Check for successful allocation */
        free(results);
        return NULL;
    }

    // Set initial conditions
    for (int i = 0; i < 6; i++) {
        Ith(y, i+1) = initial_conditions[i];
    }


    realtype t0 = time_span[0];
    realtype t1 = time_span[1];
    realtype t;
    int flag;

    // ...

    // Create and allocate CVODE memory
    void *cvode_mem = CVodeCreate(CV_BDF, CV_NORMAL);
    if (cvode_mem == NULL) { /* Check for successful allocation */
        N_VDestroy_Serial(y);
        free(results);
        return NULL;
    }

    // Initialize CVODE solver
    flag = CVodeInit(cvode_mem, repressilator_func, t0, y);
    if (flag < 0) {
        N_VDestroy_Serial(y);
        CVodeFree(&cvode_mem);
        free(results);
        return NULL;
    }

    // ...




    // Set user data
    flag = CVodeSetUserData(cvode_mem, &params);
    if (flag < 0) {
        N_VDestroy_Serial(y);
        CVodeFree(&cvode_mem);
        free(results);
        return NULL;
    }

    // Set tolerances
    flag = CVodeSStolerances(cvode_mem, 1e-6, 1e-8);
    if (flag < 0) {
        N_VDestroy_Serial(y);
        CVodeFree(&cvode_mem);
        free(results);
        return NULL;
    }

    // Loop over time points
    for (int i = 0; i < num_points; ++i) {
        t = t0 + i * (t1 - t0) / (num_points - 1);
        
        flag = CVode(cvode_mem, t, y, &t, CV_NORMAL);
        if (flag < 0) {
            N_VDestroy_Serial(y);
            CVodeFree(&cvode_mem);
            free(results);
            return NULL;
        }

        results[i*7 + 0] = t;
        for (int j = 0; j < 6; ++j) {
            results[i*7 + j + 1] = Ith(y, j+1);
        }
    }

    // Free memory
    N_VDestroy_Serial(y);
    CVodeFree(&cvode_mem);

    return results;
}

int main() {
    // Example usage
    double params_array[] = {1.0, 0.5, 2.0, 0.1, 5.0, 3.0};
    double initial_conditions[] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double time_span[] = {0.0, 100.0};
    int num_points = 1000;

    double *results = simulate_repressilator(params_array, initial_conditions, time_span, num_points);

    if (results != NULL) {
        // Output the results
        for (int i = 0; i < num_points; ++i) {
            printf("%lf ", results[i*7]);
            for (int j = 1; j <= 6; ++j) {
                printf("%lf ", results[i*7 + j]);
            }
            printf("\n");
        }
        free(results);
    }

    return 0;
}
