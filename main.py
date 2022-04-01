import imutils
import os
import numpy as np
import cv2 as cv
from glob import glob


def createDir(path):
    """
    Function to create the required directories
    If directory not exist
    Using os library
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"ERROR: Creating directory with name {path}")


def compare(frame, save, n):
    template = cv.imread('template.png')
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    template = cv.Canny(template, 10, 25)
    height, width = template.shape[:2]
    grayImage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    found = None
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resize_img = imutils.resize(grayImage, width=int(grayImage.shape[1] * scale))
        ratio = grayImage.shape[1] / float(resize_img.shape[1])
        if resize_img.shape[0] < height or resize_img.shape[1] < width:
            break
        e = cv.Canny(resize_img, 10, 25)
        match = cv.matchTemplate(e, template, cv.TM_CCOEFF)
        (_, val_max, _, loc_max) = cv.minMaxLoc(match)
        if found is None or val_max > found[0]:
            found = (val_max, loc_max, ratio)
    (_, loc_max, r) = found
    (x_start, y_start) = (int(loc_max[0]), int(loc_max[1]))
    (x_end, y_end) = (int((loc_max[0] + width)), int((loc_max[1] + height)))
    # Draw rectangle around the template
    cv.rectangle(frame, (x_start, y_start), (x_end, y_end), (153, 22, 0), 5)
    # cv.imshow('Template Found', frame)
    # cv.waitKey(0)
    # print(save)
    cv.imwrite(f"{save}/{n}.png", frame)


def frameByFrame(video, saveDir, gap=10):
    name = video.split("\\")[-1].split(".")[0]
    savePath = os.path.join(saveDir, name)
    createDir(savePath)

    capture = cv.VideoCapture(video)
    i = 1
    while True:
        ret, frame = capture.read()

        if not ret:
            capture.release()
            break
        if i == 1:
            # cv.imshow('frames', frame)
            # cv.imwrite(f"{savePath}/{i}.png", frame)
            compare(frame, savePath, i)
        else:
            if i % gap == 0:
                # cv.imshow('frames', frame)
                # cv.imwrite(f"{savePath}/{i}.png", frame)
                compare(frame, save=savePath, n=i)

        i += 1

    pass


def main():
    """
    Main function
    """
    videoPaths = glob("data/*")
    saveDir = "save"
    for path in videoPaths:
        frameByFrame(path, saveDir)


if __name__ == "__main__":
    main()
