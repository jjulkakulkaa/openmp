#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <opencv2/opencv.hpp>

int dupa() {

	int sum = 1;
#pragma omp parallel for
	for (int i = 0; i < 10; i++) {
		sum = sum * 2;
	}

	printf("%d", sum);

	return EXIT_SUCCESS;
}