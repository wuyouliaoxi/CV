import random

# pseudo-code:
# max_inliers = 0
# best_model = null
# while (i < N) {
# samples = minimal_sample(k, all_points)
# model = estimate_model(samples)
# n_inliers = computeInliers(all_points, model)
# if (n_inliers > max_inliers) {
# best_model = model
# max_inliers = n_inliers
# }
# }

def ransac(data, model, sample_size, max_iterations, threshold, ):
    max_inliers = 0
    best_model = None
    for i in range(max_iterations):
        samples = random.sample(data, sample_size)
        model = estimate(samples)
        n_inliers = 0
        for j in range(len(data)):
            pass


