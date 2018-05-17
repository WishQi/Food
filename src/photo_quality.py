import cv2
import numpy
from sklearn import svm

# hist
def hist_feature(rgb_image):
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    return cv2.calcHist([gray_image], [0], None, [256], [0.0, 256.0])

# hue, saturation, value
def hsv(rgb_image):
    HSV = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    return H, S, V

# hue, lightness, saturation
def hls(rgb_image):
    HLS = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HLS)
    H, L, S = cv2.split(HLS)
    return H, L, S

def mean_std_deviation(rgb_image):
    (mean, std_deviation) = cv2.meanStdDev(rgb_image)
    return mean, std_deviation

def contours_feature(rgb_image):
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def train_classifier(trian_data):
    clf = svm.SVC()
    clf.fit()

if __name__ == '__main__':
    image = cv2.imread('../data/food/train/bad/_5OstlynVCY1ziBlFX-8QA.jpg')
    image = cv2.resize(image, (224, 224))
    H, L, S = hls(image)
    hist = hist_feature(image)
    mean, std_deviation = mean_std_deviation(image)
    print(L)
    print(hist)
    print(mean)
    print(std_deviation)