#include "kNN.hpp"

int findLabel(int *arr, int n){
    int ele_max_freq;
    int maxCount = 0;
    for (int i = 0; i < n; i++){
        int c = 0;
        for (int j = 0; j < n; j++){
            if (arr[i] == arr[j])
                c++;
        }
        if (c > maxCount || (c == maxCount && arr[i] < ele_max_freq)){
            maxCount = c;
            ele_max_freq = arr[i];
        }
    }
    return ele_max_freq;
}

void sortArray(double array[], int array2[], int n){
    bool swapped;
    int x, y;
    for (x = 0; x < n - 1; x++){
        swapped = false;
        for (y = 0; y < n - x - 1; y++){
            if (array[y] > array[y + 1]){
                swap(array[y], array[y + 1]);
                swap(array2[y], array2[y + 1]);
                swapped = true;
            }
        }
        if (swapped == false) {
            break;
        }
    }
}

void train_test_split(Dataset &X, Dataset &Y, double test_size,
                      Dataset &X_train, Dataset &X_test, Dataset &Y_train, Dataset &Y_test){
    if (X.getData()->length() != Y.getData()->length() || test_size >= 1 || test_size <= 0) {
        return;
    }

    double minDouble = 1.0e-15;
    int nRow = X.getData()->length();
    double rowSplit = nRow * (1 - test_size);

    if (abs(round(rowSplit) - rowSplit) < minDouble * nRow) {
        rowSplit = round(rowSplit);
    }

    X_train = X.extract(0, rowSplit - 1, 0, -1);
    Y_train = Y.extract(0, rowSplit - 1, 0, -1);

    X_test = X.extract(rowSplit, -1, 0, -1);
    Y_test = Y.extract(rowSplit, -1, 0, -1);
}