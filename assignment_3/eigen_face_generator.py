from mixins import getAllImagePaths, getImages, computeEigenFaces, saveModel

PATH_TO_PROCESSED = './data/extracted_faces/'

if __name__ =='__main__':
    allImagePaths = getAllImagePaths(PATH_TO_PROCESSED, recursive=True)
    allFaces = getImages(allImagePaths)
    model = computeEigenFaces(allFaces)
    saveModel(model)
