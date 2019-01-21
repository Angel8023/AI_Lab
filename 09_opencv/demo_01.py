import cv2 as cv
import numpy as np


def rice_demo():
    src = cv.imread("D:/images/rice.png")
    src = cv.GaussianBlur(src, (3, 3), 0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("src", src)

    output = cv.connectedComponents(binary, connectivity=8, ltype=cv.CV_32S)
    num_labels = output[0]
    labels = output[1]
    colors = []
    for i in range(num_labels):
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        colors.append((b, g, r))

    colors[0] = (0, 0, 0)
    h, w = gray.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            image[row, col] = colors[labels[row, col]]

    cv.imshow("colored labels", image)
    cv.imwrite("D:/labels.png", image)
    print("total rice : ", num_labels - 1)
    cv.waitKey(0)
    cv.destroyAllWindows()


def filters_demo():
    src = cv.imread("D:/images/example.png")
    cv.imshow("input", src)
    blur = cv.blur(src, (15, 15))
    cv.imshow("blur", blur)
    gblur = cv.GaussianBlur(src, (0, 0), 15)
    cv.imshow("Gaussian Blur", gblur)

    dst = cv.bilateralFilter(src, 0, 100, 15)
    cv.imshow("Bilateral Filter", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


def binary_demo():
    src = cv.imread("D:/images/text1.png")
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    ada_ = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
    cv.imshow("binary", binary)
    cv.imshow("ada_", ada_)
    print("threshold : %.2f"%ret)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    rice_demo()
