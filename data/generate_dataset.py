import sys

import numpy as np

def crop_points(points):
    points = points[np.all(points <= 10, axis=-1)]
    points = points[np.all(points >= 0, axis=-1)]
    return points

def generate_points(dim, mean, cov, N):
    points = np.empty((0, dim))

    while points.shape[0] < N:
        n = N - points.shape[0]
        points = np.concatenate((points, np.random.multivariate_normal(mean, cov, n)))
        points = crop_points(points)
    return points


def generate_dataset1(dim, name):
    mean = np.ones(dim)
    cov = np.eye(dim,dim)
    N = 10+40*(dim-1)

    points = generate_points(dim, mean, cov, N)
    np.save(name, points)
    return points

def generate_dataset2(dim, name):
    mean = np.ones(dim)
    cov = np.eye(dim,dim)
    N = 5+30*(dim-1)

    points = generate_points(dim, mean, cov, N)
    for d in range(dim):
        mean[d] = 9
        p = generate_points(dim, mean, cov, N)
        points = np.concatenate((points, p))
        mean[d] = 1

    np.save(name, points)
    return points

def generate_dataset3(dim, name):
    mean = np.ones(dim)*5
    if(dim>1):
        cov = 9*np.eye(dim, dim) + ((-(9/(dim-1)) + 0.5) * (1-np.eye(dim, dim)))# + np.array([[9, -8.9], [-8.9, 9]]) #8.5
    else:
        cov = np.eye(dim, dim)*0.5
    N = 5+50*((dim-1)**2)

    points = generate_points(dim, mean, cov, N)

    np.save(name, points)
    return points

if __name__ == "__main__":
    set = [generate_dataset1, generate_dataset2, generate_dataset3]

    generator = set[int(sys.argv[1])]
    d = int(sys.argv[2])
    name = sys.argv[3]

    if name[-4:] != ".npy":
        name += ".npy"

    p = generator(d, name)