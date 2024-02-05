#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>

typedef struct {
    double deg_m, deg_p, alpha, alpha_0, beta, n;
} repressilator_params;

int repressilator_func(double t, const double y[], double f[], void *params) {
    (void)(t);
    repressilator_params *p = (repressilator_params *)params;
    double deg_m = p->deg_m;
    double deg_p = p->deg_p;
    double alpha = p->alpha;
    double alpha_0 = p->alpha_0;
    double beta = p->beta;
    double n = p->n;

    f[0] = -deg_m * y[0] + alpha / (1 + pow(y[5], n)) + alpha_0; 
    f[1] = -deg_m * y[1] + alpha / (1 + pow(y[3], n)) + alpha_0; 
    f[2] = -deg_m * y[2] + alpha / (1 + pow(y[4], n)) + alpha_0; 

    f[3] = -deg_p * y[3] + beta * y[0]; 
    f[4] = -deg_p * y[4] + beta * y[1];
    f[5] = -deg_p * y[5] + beta * y[2];

    return GSL_SUCCESS;
}

double* simulate_repressilator(double *params_array, double *initial_conditions, double *time_span, double num_points) {
    int numi = (int) num_points;

    gsl_odeiv2_system sys = {repressilator_func, NULL, 6, params_array};

    double t0 = time_span[0];
    double t1 = time_span[1];

    double span = time_span[1]-time_span[0];

    double h = 1e-6;
    gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, h, 1e-8, 0.0);

    double* results = malloc(numi * 7 * sizeof(double));
    if (!results) {
        fprintf(stderr, "Failed to allocate memory for results\n");
        return NULL;
    }

    int i;
    double ti;
    
    for (i = 0; i < numi; i=i+1) {
        ti = t0 + (span / (num_points));
        
        int status = gsl_odeiv2_driver_apply(d, &t0, ti, initial_conditions);

        if (status != GSL_SUCCESS) {
            printf("Error, return value=%d\n", status);
            free(results);
            return NULL;
        }

        results[(i * 7) + 0] = ti;
        results[(i * 7) + 1] = initial_conditions[0];
        results[(i * 7) + 2] = initial_conditions[1];
        results[(i * 7) + 3] = initial_conditions[2];
        results[(i * 7) + 4] = initial_conditions[3];
        results[(i * 7) + 5] = initial_conditions[4];
        results[(i * 7) + 6] = initial_conditions[5];
    }

    gsl_odeiv2_driver_free(d);

    return results;
}
