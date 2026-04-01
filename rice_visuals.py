import cv2
import numpy as np
from matplotlib import pyplot as plt
from flask import Flask, request, render_template

import matplotlib
matplotlib.use('Agg')  # Add this at the top
import matplotlib.pyplot as plt


def label_image(img, contours):
    labeled_img = img.copy()
    for i, contour in enumerate(contours):
        cv2.drawContours(labeled_img, [contour], -1, (0,255,0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(labeled_img, str(i+1), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    return labeled_img

def plot_histogram(df, column, fname):
    plt.hist(df[column])
    plt.title(f"{column} Distribution")
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.savefig(fname)
    plt.close()

def plot_bar_chart(df, fname):
    df['Category'].value_counts().plot(kind='bar')
    plt.title('Rice Type Count')
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.savefig(fname)
    plt.close()
