import random
import math
def foo(n):
    random.seed()
    c1 = 0
    c2 = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        r1 = x * x + y * y
        r2 = (1 - x) * (1 - x) + (1 - y) * (1 - y)
        if r1 <= 1 and r2 <= 1:
            c1 += 1
        else:
            c2 += 1
    return c1 / c2

def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA**0.5)*(normB**0.5)) * 100, 2)


def main():
    vector1 = [1,2,0,2,1]
    vector2 = [1,3,0,1,3]
    vector3 = [0,2,0,1,1]
    print(cosine_similarity(vector1,vector2),cosine_similarity(vector1,vector3))
    print(math.e ** (6 / 21))#1.3299852842789415

if __name__ == "__main__":
    main()
