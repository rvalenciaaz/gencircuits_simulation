#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

    return 0;
}

void euler_maruyama(repressilator_params *params, double y[], double dt, double noise_strength) {
    double f[6];
    repressilator_func(0, y, f, params);

    for (int i = 0; i < 6; i++) {
        double noise = noise_strength * sqrt(dt) * ((double) rand() / RAND_MAX - 0.5);
        y[i] += f[i] * dt + noise;
    }
}

double* simulate_repressilator(repressilator_params *params, double *initial_conditions, double *time_span, int num_points) {
    double dt = (time_span[1] - time_span[0]) / num_points;
    double noise_strength = 0.1; // Adjust this value as needed for your simulation

    double* results = malloc(num_points * 7 * sizeof(double));
    if (!results) {
        fprintf(stderr, "Failed to allocate memory for results\n");
        return NULL;
    }

    double y[6];
    for (int i = 0; i < 6; i++) {
        y[i] = initial_conditions[i];
    }

    for (int i = 0; i < num_points; i++) {
        double t = time_span[0] + i * dt;
        euler_maruyama(params, y, dt, noise_strength);

        results[i * 7 + 0] = t;
        for (int j = 0; j < 6; j++) {
            results[i * 7 + 1 + j] = y[j];
        }
    }

    return results;
}
