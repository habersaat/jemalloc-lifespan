#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <jemalloc/jemalloc.h>

#define EPOCHS 50
#define LR 0.01


#define N_FEATURES 20000  
#define N_SAMPLES 20000 
#define PAD_SIZE 20000  

float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

int main() {
    srand((unsigned int)time(NULL));

    float **X = je_malloc(N_SAMPLES * sizeof(float *));
    float *y = je_malloc(PAD_SIZE * sizeof(float));
    float *weights = je_malloc(PAD_SIZE * sizeof(float));

    for (int i = 0; i < N_SAMPLES; i++) {
        X[i] = je_malloc(PAD_SIZE * sizeof(float));

        float sum = 0.0f;
        for (int j = 0; j < N_FEATURES; j++) {
            X[i][j] = (float)rand() / RAND_MAX;
            if (j < 5) sum += X[i][j];
        }
        y[i] = sum > 2.5f ? 1 : 0;
    }

    for (int j = 0; j < N_FEATURES; j++)
        weights[j] = 0.0f;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float *grads = je_malloc(PAD_SIZE * sizeof(float));

        for (int j = 0; j < N_FEATURES; j++) grads[j] = 0.0f;

        int correct = 0;
        for (int i = 0; i < N_SAMPLES; i++) {
            float z = 0.0f;
            for (int j = 0; j < N_FEATURES; j++)
                z += weights[j] * X[i][j];

            float pred = sigmoid(z);
            float error = pred - y[i];

            int predicted_label = (pred >= 0.5f) ? 1 : 0;
            if (predicted_label == (int)y[i]) correct++;

            for (int j = 0; j < N_FEATURES; j++)
                grads[j] += error * X[i][j];
        }

        for (int j = 0; j < N_FEATURES; j++)
            weights[j] -= LR * grads[j] / N_SAMPLES;

        je_free(grads);

        float acc = 100.0f * correct / N_SAMPLES;
        printf("Epoch %d complete – Training Accuracy: %.2f%%\n", epoch + 1, acc);
    }

    for (int i = 0; i < N_SAMPLES; i++) je_free(X[i]);
    je_free(X); je_free(y); je_free(weights);

    return 0;
}
