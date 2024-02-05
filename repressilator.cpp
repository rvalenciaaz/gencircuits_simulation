#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>

struct RepressilatorParams {
    double deg_m, deg_p, alpha, alpha_0, beta, n;
};

int repressilator_func(double t, const double y[], double f[], void *params) {
    (void)(t); // Unused parameter
    RepressilatorParams *p = static_cast<RepressilatorParams *>(params);

    double deg_m = p->deg_m;
    double deg_p = p->deg_p;
    double alpha = p->alpha;
    double alpha_0 = p->alpha_0;
    double beta = p->beta;
    double n = p->n;

    f[0] = -deg_m * y[0] + alpha / (1 + std::pow(y[5], n)) + alpha_0; 
    f[1] = -deg_m * y[1] + alpha / (1 + std::pow(y[3], n)) + alpha_0; 
    f[2] = -deg_m * y[2] + alpha / (1 + std::pow(y[4], n)) + alpha_0; 

    f[3] = -deg_p * y[3] + beta * y[0]; 
    f[4] = -deg_p * y[4] + beta * y[1];
    f[5] = -deg_p * y[5] + beta * y[2];

    return GSL_SUCCESS;
}

double* simulate_repressilator(double *params_array, double *initial_conditions, double *time_span, double num_points) {
    int numi = static_cast<int>(num_points);

    gsl_odeiv2_system sys = {repressilator_func, nullptr, 6, params_array};

    double t0 = time_span[0];
    double t1 = time_span[1];

    double span = t1 - t0;

    double h = 1e-6;
    gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, h, 1e-8, 0.0);

    double* results = static_cast<double*>(malloc(numi * 7 * sizeof(double)));
    if (!results) {
        std::fprintf(stderr, "Failed to allocate memory for results\n");
        return nullptr;
    }

    for (int i = 0; i < numi; ++i) {
        double ti = t0 + (span / num_points);
        
        int status = gsl_odeiv2_driver_apply(d, &t0, ti, initial_conditions);

        if (status != GSL_SUCCESS) {
            std::printf("Error, return value=%d\n", status);
            free(results);
            return nullptr;
        }

        for (int j = 0; j < 7; ++j) {
            results[i * 7 + j] = (j == 0) ? ti : initial_conditions[j - 1];
        }
    }

    gsl_odeiv2_driver_free(d);

    return results;
}
