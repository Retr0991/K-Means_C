#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define max_iterations 1000
#define number_of_runs 50
#define no_of_data_point 150
#define no_of_attribute 4
#define k 3
#define ADDRESS "IRIS_input-dataset.csv"

// returns the euclidian distance between 2 data point
float distance(float *arr, float *cl)
{
    float res = 0;
    for (int i = 1; i < no_of_attribute + 1; ++i)
    {
        res = res + pow((arr[i] - cl[i - 1]), 2);
    }
    res = sqrt(res);
    return res;
}

// function to assign cluster info in the first element of any data point
void assign_cluster_info(float **arr, float **c_info)
{
    for (int i = 0; i < no_of_data_point; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            arr[i][0] = (distance(arr[i], c_info[(int)arr[i][0]]) > distance(arr[i], c_info[j])) ? j : arr[i][0];
        }
    }
}

// function to change the centroids
void update_centroid(float **arr, float **centroidal_data)
{
    int cnt;
    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < no_of_attribute + 1; ++j)
        {
            centroidal_data[i][j] = 0;
        }
    }
    for (int i = 0; i < k; ++i)
    {
        cnt = 0;
        for (int j = 0; j < no_of_data_point; ++j)
        {
            if ((int)arr[j][0] == i)
            {
                for (int t = 1; t < no_of_attribute + 1; t++)
                {
                    centroidal_data[i][t - 1] += arr[j][t];
                }
                cnt++;
            }
        }
        if (cnt != 0)
        {
            for (int t = 0; t < no_of_attribute + 1; t++)
            {
                centroidal_data[i][t] /= (float)cnt;
            }
        }
    }
}

bool read_dataset_from_file(float **attribute_arr)
{
    FILE *file;
    char buffer[100]; // Buffer to read each line from the file
    int row = 0, col = 0;

    file = fopen(ADDRESS, "r");
    if (file == NULL)
    {
        printf("Error opening the file.\n");
        return false;
    }

    // read each line from the file and parse the comma-separated values
    while (fgets(buffer, sizeof(buffer), file) != NULL)
    {
        // Tokenize the line based on commas
        char *token;
        token = strtok(buffer, ",");
        col = 1;

        // Convert the tokens to floating-point values and store them in attribute_arr
        while (token != NULL)
        {
            attribute_arr[row][col] = atof(token);
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }

    fclose(file);
    return true;
}

int randomly_select_first_centroid()
{
    srand((unsigned int)time(NULL));
    return rand() % no_of_data_point;
}

// randomly initialises the centroids... better than forgy method
void kmeans_plus_plus_init(float **data_points, float **centroidal_data)
{
    int first_centroid_index = randomly_select_first_centroid();
    for (int j = 0; j < no_of_attribute; j++)
    {
        centroidal_data[0][j] = data_points[first_centroid_index][j + 1];
    }

    for (int i = 1; i < k; i++)
    {
        // Calculate the squared distances to the nearest centroid for each data point
        float *distances_sq = (float *)malloc(no_of_data_point * sizeof(float));
        float sum_dist_sq = 0;

        for (int j = 0; j < no_of_data_point; j++)
        {
            float min_dist_sq = distance(data_points[j], centroidal_data[0]);
            for (int c = 1; c < i; c++)
            {
                float dist_sq = distance(data_points[j], centroidal_data[c]);
                min_dist_sq = fmin(min_dist_sq, dist_sq);
            }
            distances_sq[j] = min_dist_sq;
            sum_dist_sq += min_dist_sq;
        }

        // Choose the next centroid randomly with probability proportional to the squared distance
        float rand_val = (float)rand() / RAND_MAX;
        float prob_sum = 0;
        for (int j = 0; j < no_of_data_point; j++)
        {
            prob_sum += distances_sq[j] / sum_dist_sq;
            if (rand_val <= prob_sum)
            {
                for (int t = 0; t < no_of_attribute; t++)
                {
                    centroidal_data[i][t] = data_points[j][t + 1];
                }
                break;
            }
        }
        free(distances_sq);
    }
}

// running the algorithm for multiple times
void kmeans_multiple_runs(float **attribute_arr, float **centroidal_data, int num_runs)
{
    float **best_centroids = (float **)calloc(k, sizeof(float *));
    // initializing best_sum to inf and from there it goes on decreasing
    float best_sum = INFINITY;

    for (int run = 0; run < num_runs; run++)
    {
        // Initialize the centroids using K-means++
        kmeans_plus_plus_init(attribute_arr, centroidal_data);

        for (int i = 0; i < max_iterations; i++)
        {
            assign_cluster_info(attribute_arr, centroidal_data);
            update_centroid(attribute_arr, centroidal_data);
        }

        // calculate the euclidean distances for the current run
        float current_sum = 0.0;
        for (int i = 0; i < no_of_data_point; i++)
        {
            int cluster_id = (int)attribute_arr[i][0];
            current_sum += distance(attribute_arr[i], centroidal_data[cluster_id]);
        }

        // if the current result is smol, update the best result
        if (current_sum < best_sum)
        {
            best_sum = current_sum;
            for (int i = 0; i < k; i++)
            {
                if (best_centroids[i] == NULL)
                {
                    best_centroids[i] = (float *)calloc(no_of_attribute, sizeof(float));
                }
                memcpy(best_centroids[i], centroidal_data[i], (no_of_attribute) * sizeof(float));
            }
        }
    }
    for (int i = 0; i < k; i++)
    {
        memcpy(centroidal_data[i], best_centroids[i], (no_of_attribute) * sizeof(float));
    }
    // free memory
    for (int i = 0; i < k; i++)
    {
        free(best_centroids[i]);
    }
    free(best_centroids);
}

// driver code
int main()
{
    // memory allocations using calloc inplace of malloc as it initialises the elements to zero and is contiguous
    float **centroidal_data = (float **)calloc(k, sizeof(float *));
    for (int i = 0; i < k; i++)
    {
        centroidal_data[i] = (float *)calloc((no_of_attribute), sizeof(float));
    }
    float **attribute_arr = (float **)calloc(no_of_data_point, sizeof(float *));
    for (int i = 0; i < no_of_data_point; ++i)
    {
        attribute_arr[i] = (float *)calloc(no_of_attribute + 1, sizeof(float));
    }

    if (!read_dataset_from_file(attribute_arr))
        return 0;

    kmeans_multiple_runs(attribute_arr, centroidal_data, number_of_runs);

    // this line was meant to print the cluster info but decided to remove it as there are too many data points
    // for (int i = 0; i < no_of_data_point; ++i)
    // {
    //     printf("%0.0f\n",attribute_arr[i][0]);
    // }
    
    // print centroids
    printf("Final Centroids:\n\n");
    for (int i = 0; i < k; ++i)
    {
        printf("Centroid %d: ", i + 1);
        for (int j = 0; j < no_of_attribute; ++j)
        {
            printf("%.4f ", centroidal_data[i][j]);
        }
        printf("\n");
    }

    // free memory
    for (int i = 0; i < k; i++)
    {
        free(centroidal_data[i]);
    }
    free(centroidal_data);

    for (int i = 0; i < no_of_data_point; i++)
    {
        free(attribute_arr[i]);
    }
    free(attribute_arr);
    return 0;
}

