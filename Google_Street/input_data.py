import numpy as np
import cv2 as cv
import csv

def LoadImageData( DataNum, Size ):

    ImageData = np.empty(shape=(DataNum, Size, Size, 3))

    for ImageNum in range(DataNum):
        ImageName = './DataSet/train/%d.Bmp' % (ImageNum+1)
        SourceImage = cv.imread(ImageName)
        ResizedImage = cv.resize(SourceImage, (Size, Size))
        ImageData[ImageNum] = ResizedImage/255.0

    return ImageData, np.shape(ImageData)


def LoadLabels( DataNum ):
    LabelCount = 0
    LabelData = np.zeros(shape=[DataNum, 36] )
    with open("./DataSet/trainLabels.csv", "rb" ) as label:
        LabelReader = csv.reader(label)
        for i in LabelReader:
            if ( len(i[1]) > 1 ):
                continue
            if ( ord(i[1]) <= ord('9') ):
                LabelData[LabelCount, (ord(i[1])-ord('0'))] = 1
            elif ( ord(i[1]) >= ord('a')):
                LabelData[LabelCount, (ord(i[1])-ord('a')+10)] = 1
            elif (ord(i[1]) >= ord('A')):
                LabelData[LabelCount, (ord(i[1])-ord('A')+10)] = 1

            LabelCount = LabelCount + 1
            if ( LabelCount > DataNum-1 ):
                break

    return LabelData, np.shape(LabelData)



# def LoadImageData( DataNum, Size ):
#     Image = np.empty(shape=(3, Size, Size))
#     ImageData = []
#
#     for ImageNum in range(DataNum):
#         ImageName = './DataSet/train/%d.Bmp' % (ImageNum+1)
#         SourceImage = cv.imread(ImageName)
#         ResizedImage = cv.resize(SourceImage, (Size, Size))/255.0
#         Image = np.array([ResizedImage[:, :, 0], ResizedImage[:, :, 1], ResizedImage[:, :, 2]])
#         ImageData.append(Image)
#
#     return ImageData, np.shape(ImageData)


# def LoadImageData2( DataNum, Width, Height ):
#
#     ImageData = np.empty(shape=(DataNum, Width, Height, 3))
#
#     for ImageNum in range(DataNum):
#         ImageName = './DataSet/train/%d.Bmp' % (ImageNum+1)
#         SourceImage = cv.imread(ImageName)
#         ResizedImage = cv.resize(SourceImage, (Width, Height))
#         ImageData[ImageNum] = ResizedImage/255.0
#
#     return ImageData, np.shape(ImageData)


def Reshape( Image ):
    ImageShape = np.shape(Image)
    ReshapedImage = np.empty([ImageShape[1], ImageShape[2], ImageShape[0]], dtype=float)
    ReshapedImage[:, :, 0] = Image[0, :, :]
    ReshapedImage[:, :, 1] = Image[1, :, :]
    ReshapedImage[:, :, 2] = Image[2, :, :]
    return ReshapedImage