from PIL import Image, ImageDraw
import numpy as np
from skimage.metrics import structural_similarity
import cv2
import matplotlib.pyplot as plt

MAX_ITERATIONS = 200
MARKER_POSITION = 0.5

class TestImage:

    def __init__(self, imgPath, polySize):
        """
        Initializes an instance of the class
        :param imgPath: path of the file containing the reference image
        :param polySize: number of vertices on the polygons used to recreate the image
        """
        self.refImg = Image.open(imgPath)
        self.polySize = polySize

        self.width, self.height = self.refImg.size
        self.numPixels = self.width * self.height
        self.refImgCv2 = self.toCv2(self.refImg)

    def polygonDataToImage(self, polyData):
        """
        Accepts polygon data and creates an image containing these polygons.
        :param polyData: a list of polygon parameters. Each item in the list
        represents the vertices locations, color, and transparency of the corresponding polygon
        :return: the image containing the polygons (Pillow format)
        """

        # Start with a new image:
        image = Image.new('RGB', (self.width, self.height))
        draw = ImageDraw.Draw(image, 'RGBA')

        # Divide the polyData into chunks, each containing the data for a single polygon:
        chunkSize = self.polySize * 2 + 4  # (x,y) per vertex + (RGBA)
        polygons = self.splitList(polyData, chunkSize)

        # Iterate over all polygons and draw each of them into the image:
        for poly in polygons:
            index = 0

            # Extract the vertices of the current polygon:
            vertices = []
            for vertex in range(self.polySize):
                vertices.append((int(poly[index] * self.width), int(poly[index + 1] * self.height)))
                index += 2

            # Extract the RGB and alpha values of the current polygon:
            red = int(poly[index] * 255)
            green = int(poly[index + 1] * 255)
            blue = int(poly[index + 2] * 255)
            alpha = int(poly[index + 3] * 255)

            # Draw the polygon into the image:
            draw.polygon(vertices, (red, green, blue, alpha))

        # Cleanup:
        del draw

        return image

    def getDifference(self, polyData, method="MSE"):
        """
        Accepts polygon data, creates an image containing these polygons, and calculates the difference
        between this image and the reference image using one of two methods.
        :param polyData: a list of polygon parameters. Each item in the list
        represents the vertices locations, color, and transparency of the corresponding polygon
        :param method: base method of calculating the difference ("MSE" or "SSIM").
        Larger return value always means larger difference
        :return: the calculated difference between the image containing the polygons and the reference image
        """

        # Create the image containing the polygons:
        image = self.polygonDataToImage(polyData)

        if method == "MSE":
            return self.calculateMse(image)
        else:
            return 1.0 - self.calculateSsim(image)

    def plotImages(self, image, header=None):
        """
        Creates a 'side-by-side' plot of the given image next to the reference image
        :param image: image to be drawn next to reference image (Pillow format)
        :param header: text used as a header for the plot
        """

        fig = plt.figure("Image Comparison:")
        if header:
            plt.suptitle(header)

        # Plot the reference image on the left:
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(self.refImg)
        self.hideTicks(plt)

        # Plot the given image on the right:
        fig.add_subplot(1, 2, 2)
        plt.imshow(image)
        self.hideTicks(plt)

        return plt

    def saveImage(self, polyData, imgFilePath, header=None):
        """
        Accepts polygon data, creates an image containing these polygons,
        creates a 'side-by-side' plot of this image next to the reference image,
        and saves the plot to a file
        :param polyData: a list of polygon parameters. Each item in the list
        represents the vertices locations, color, and transparency of the corresponding polygon
        :param imgFilePath: path of file to be used to save the plot to
        :param header: text used as a header for the plot
        """

        # Create an image from the polygon data:
        image = self.polygonDataToImage(polyData)

        # Plot the image side-by-side with the reference image:
        self.plotImages(image, header)

        # Save the plot to file:
        plt.savefig(imgFilePath)

    # Utility methods:

    def toCv2(self, pilImage):
        """Converts the given Pillow image to CV2 format"""
        return cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGB2BGR)

    def calculateMse(self, image):
        """Calculates MSE of difference between the given image and the reference image"""
        return np.sum((self.toCv2(image).astype("float") - self.refImgCv2.astype("float")) ** 2) / float(self.numPixels)

    def calculateSsim(self, image):
        """Calculates mean structural similarity index between the given image and the reference image"""
        return structural_similarity(self.toCv2(image), self.refImgCv2, multichannel=True)

    def splitList(self, list, chunkSize):
        """Divides a given list into fixed size chunks, returns a generator iterator"""
        for chunk in range(0, len(list), chunkSize):
            yield(list[chunk:chunk + chunkSize])

    def hideTicks(self, plot):
        """Turns off ticks on both axes"""
        plt.tick_params(
            axis='both',
            which='both',
            bottom=False,
            left=False,
            top=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
