import numpy as np
from plyfile import PlyData, PlyElement


def open_ply(file_path):
    rdata = PlyData.read(file_path)
    points = []
    for i in range(len(rdata.elements[0].data)):
        point = rdata.elements[0].data[i]
        a = np.array(list(point))
        points.append(a)
    data = np.array(points)
    return data


def write_ply(name, data):
    tuples = []
    for point_i in range(data.shape[0]):
        tuples.append(tuple(data[point_i, :9]))

    described_data = np.array(
        tuples,
        dtype=[
            ("x", "double"),
            ("y", "double"),
            ("z", "double"),
            ("nx", "double"),
            ("ny", "double"),
            ("nz", "double"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    element = PlyElement.describe(described_data, "vertex")
    PlyData([element], text=False).write(name)


def main():
    points = open_ply("data/PointCloud.ply")
    # Your code goes here

    write_ply("data/Result.ply", points)
    return 0


if __name__ == "__main__":
    main()
