import matplotlib.pyplot as plt

def subtract(p0, p1):
    return (p0[0] - p1[0], p0[1], p1[1])

def dotProduct(p0, p1):
    return (p0[0] * p1[0]) + (p0[1] * p1[1])

def isInside(triangle, point):
    p0, p1, p2 = triangle[0], triangle[1], triangle[2]
    dxdy = subtract(p1, p0)
    normal = (dxdy[1], -dxdy[0])
    line1_check = dotProduct(subtract(point, p0), normal) > 0
    print(line1_check)

    dxdy = subtract(p2,p1)
    normal = (dxdy[1], -dxdy[0])
    line2_check = dotProduct(subtract(point, p1), normal) <0
    print(line2_check)

    dxdy = subtract(p0,p2)
    normal = (dxdy[1], -dxdy[0])
    line3_check = dotProduct(subtract(point, p2), normal) > 0
    print(line3_check)

    return line1_check and line2_check and line3_check

triangle = [
    (3, 1),
    (2, 5),
    (7, 3),
]

inside_point = (3,2)
outside_point = (2,1)

points = [inside_point, outside_point]

for point in points:
    color = 'go' if isInside(triangle, point) else 'ro'
    plt.plot([point[0]], [point[1]], color)

plt.plot([triangle[0][0], triangle[1][0]], [triangle[0][1], triangle[1][1]], 'b')
plt.plot([triangle[1][0], triangle[2][0]], [triangle[1][1], triangle[2][1]], 'b')
plt.plot([triangle[0][0], triangle[2][0]], [triangle[0][1], triangle[2][1]], 'b')
plt.show()