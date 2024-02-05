#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>

typedef struct {
    double deg_m, deg_p, alpha, alpha_0, beta, n, cyc;
} repressilator_params;

int repressilator_func(double t, const double y[], double f[], void *params) {
    (void)(t);
    int k;
    int d;
    repressilator_params *p = (repressilator_params *)params;
    double deg_m = p->deg_m;
    double deg_p = p->deg_p;
    double alpha = p->alpha;
    double alpha_0 = p->alpha_0;
    double beta = p->beta;
    double n = p->n;
    int cyc = p->cyc;

    for (k = 0; k < cyc; k=k+1) {
    if (k==0){    
    f[k] = -deg_m * y[k] + alpha / (1 + pow(y[2*cyc-1-k], n)) + alpha_0; 
    }
    else{
    f[k] = -deg_m * y[k] + alpha / (1 + pow(y[cyc+k-1], n)) + alpha_0;
    } 
    }

    //2*cyc=6
    //cyc=3
    for (d = 0; d < cyc; d=d+1) {

    f[2*cyc-1-d] = -deg_p * y[2*cyc-1-d] + beta * y[cyc-1-d];

    }

    return GSL_SUCCESS;
}

double* simulate_repressilator(double *params_array, double *initial_conditions, double *time_span, double num_points, int cyc) {
    int numi = (int) num_points;

    gsl_odeiv2_system sys = {repressilator_func, NULL, cyc*2, params_array};

    double t0 = time_span[0];
    double t1 = time_span[1];

    double span = time_span[1]-time_span[0];

    double h = 1e-6;
    gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, h, 1e-8, 0.0);

    double* results = malloc(numi * (2*cyc + 1) * sizeof(double));
    if (!results) {
        fprintf(stderr, "Failed to allocate memory for results\n");
        return NULL;
    }

    int i;
    int q;
    double ti;
    
    for (i = 0; i < numi; i=i+1) {
        ti = t0 + (span / (num_points));
        
        int status = gsl_odeiv2_driver_apply(d, &t0, ti, initial_conditions);

        if (status != GSL_SUCCESS) {
            printf("Error, return value=%d\n", status);
            free(results);
            return NULL;
        }

        for (q = 0; q < (2*cyc + 1); q=q+1) {

        if (q==0){
            results[(i * (2*cyc + 1)) + 0] = ti;
        }
        else{
            results[(i * (2*cyc + 1)) + q] = initial_conditions[q-1];
        }
        
        }
    }

    gsl_odeiv2_driver_free(d);

    return results;
}
