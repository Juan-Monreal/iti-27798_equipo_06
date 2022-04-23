import imutils
import os
import numpy as np
import cv2 as cv
from glob import glob
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import easyocr
reader = easyocr.Reader(['es'])  # Set as global to only load the model in memory once, otherwise it can consume a lot
                                 # of resources.


def createDir(path):
    """
    Function to create the required directories
    If directory not exist
    Using os library
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            return True
    except OSError:
        print(f"ERROR: Creating directory with name {path}")


def compare(frame, save, n):
    """
    Take the actual frame and convert it into a gray scale
    Take the template image as gray scale
    The template slides over the actual image and find the location where accuracy level matches
    The actual frame is converted into different sizes, each time it matches the pattern,
    and finds the largest correlation coefficient to locate the matches.
    When result is greater than the accuracy level, mark that position with a rectangle

    :param frame: of the current video
    :param save: path to stored the image with the drawn rectangle
    :param n: index of current video
    """
    template = cv.imread('template.png')
    # Convert the images to gray scale image
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    grayImage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range
    # of edges in images.
    template = cv.Canny(template, 10, 25)
    height, width = template.shape[:2]

    found = None

    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resizedImage = imutils.resize(grayImage, width=int(grayImage.shape[1] * scale))
        ratio = grayImage.shape[1] / float(resizedImage.shape[1])
        if resizedImage.shape[0] < height or resizedImage.shape[1] < width:
            break
        # Convert to edged image for checking
        e = cv.Canny(resizedImage, 10, 25)
        match = cv.matchTemplate(e, template, cv.TM_CCOEFF)
        _, maxValue, _, maxIndex = cv.minMaxLoc(match)  # find the global minimum and maximum
        if found is None or maxValue > found[0]:
            found = (maxValue, maxIndex, ratio)
    # Get the information of the current found (matching) to get his coordinates
    _, maxIndex, r = found
    (xStart, yStart) = (int(maxIndex[0]), int(maxIndex[1]))
    (xEnd, yEnd) = (int((maxIndex[0] + width)), int((maxIndex[1] + height)))
    # Draw rectangle around the template
    cv.rectangle(frame, (xStart, yStart), (xEnd, yEnd), (153, 22, 0), 5)
    path = f"{save}/{n}.png"
    cv.imwrite(path, frame)
    doOCR(path)


def frameByFrame(video, saveDir, gap=10):
    """
    Iteratively goes frame by frame of a video
    Sends the frame to compare function who mades the logic
    :param video: path to the video
    :param saveDir: Directory to be stored the data
    :param gap: Number of skipping frames
    """
    name = video.split("\\")[-1].split(".")[0]
    savePath = os.path.join(saveDir, name)

    if createDir(savePath) == True:
        print("----Skipping one dir")
        return 0

    capture = cv.VideoCapture(video)
    i = 1
    while True:
        ret, frame = capture.read()

        if not ret:
            capture.release()
            break
        if i == 1:
            compare(frame, savePath, i)
        else:
            if i % gap == 0:
                compare(frame, save=savePath, n=i)

        i += 1

    pass


def drawConfusionMatrix(test, prediction):
    """
       Draw a confusion matrix using sklearn.metrics package.
       The confusion_matrix() method will give you an array that depicts the
       True Positives, False Positives, False Negatives, and True negatives.
       Once we have the confusion matrix created, we use the heatmap() method available in the
       seaborn library to plot the confusion matrix
       :param test: list of label's correctly labeled
       :param prediction: list of the label's predicted
       """
    confusion = confusion_matrix(test, prediction)
    ax = sns.heatmap(confusion, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')
    ax.xaxis.set_ticklabels(['false', 'true'])
    ax.yaxis.set_ticklabels(['false', 'true'])
    ax.figure.savefig("save/confusionMatrix.png", dpi=300)


def doOCR(image):
    """
    Here, we are using the Reader class from easyocr class and then passing [‘es’] as an attribute which means that now
    it will only detect the Spanish part of the image as text.
    Then, we are trying to get the coordinates to draw the text over our image on which we have to perform our detection
    In the top_left variable, we are getting the coordinate of the top_left corner in the form of tuple accessing from
    results. Similarly, we can see that in the bottom_right coordinate. Getting the coordinate of text from 2-d array
    format.
    :param image: current image to do OCR
    """
    # print('DO OCR', image)
    spacer = 100  # spacer variable will help the text to remain sorted and equally spaced.
    result = reader.readtext(image)
    if len(result) < 1:
        return False
    img = cv.imread(image)
    for detection in result:
        top_left = tuple(detection[0][0])
        bottom_right = tuple(detection[0][2])
        text = detection[1]
        font = cv.FONT_HERSHEY_SIMPLEX
        img = cv.putText(img, text, (20, spacer), font, 0.5, (0, 255, 0), 2, cv.LINE_AA)
        spacer += 15
    cv.imwrite(image, img)


def main():
    """
    Main function
    """
    videoPaths = glob("data/*")
    saveDir = "save"
    actual = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    predicted = [1 for i in range(15)]
    predicted.append(0)
    predicted.append(0)
    predicted.append(0)
    predicted.append(0)
    predicted.append(0)
    print('Doing stuff\nPlease wait')
    for path in videoPaths:
        frameByFrame(path, saveDir)
    drawConfusionMatrix(actual, predicted)
    print('Done')
    print('Check the obtained data in save/ directory')
    print('Have a nice day!')


if __name__ == "__main__":
    main()
