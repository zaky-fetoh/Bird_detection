import cv2
import numpy as np


# cap = cv2.VideoCapture(0) #for automatic USBCam (0-web cam default)


def getbirds(img):
    img = np.array(img)

    img1 = np.concatenate([img[np.newaxis, 2:, 2:],
                           img[np.newaxis, :-2, :-2],
                           # img[np.newaxis, 1:-1, 1:-1],
                           # img[np.newaxis, 1:-1, 2:],
                           # img[np.newaxis, 1:-1, :-2],
                           # img[np.newaxis, 2:, 1:-1],
                           img[np.newaxis, :-2, 1:-1], ])
    img = np.amin(img1, axis=0)
    image_res, image_thresh = cv2.threshold(img, 0, 255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    image_thresh = 255 - image_thresh
    # Find total markers
    dist_8u = image_thresh.astype('uint8')
    # Find total markers
    contours, _ = cv2.findContours(dist_8u,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # Create the marker image for the watershed algorithm
    out, in_chem, chem, chem_indx, chem_cont = list(), list(), -1, None, None
    # Draw the foreground markers
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        out.append((x, y, w, h))
        if chem < w * h:
            chem = w * h
            chem_cont = c
            chem_indx = out.__len__() - 1
    del out[chem_indx]
    for x, y, w, h in out:
        result = cv2.pointPolygonTest(chem_cont,
                                      (x + w / 2, y + h + 10),
                                      False)
        if result != -1:
            in_chem.append(result)
    return out, in_chem.__len__()


def putxt(img, txt, org=(0, 25)):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org

    # fontScale
    fontScale = .5

    # Blue color in BGR
    color = (0, 255, 0)

    # Line thickness of 2 px
    thickness = 1
    return cv2.putText(img, txt, org, font,
                       fontScale, color, thickness, cv2.LINE_AA)


class DetectBirds(object):
    def __init__(self, camera_url, ):
        self.cap = cv2.VideoCapture(camera_url)
        self.running = True

    def detect(self):
        tot = 0
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                birds, inchem = getbirds(gray)
                tot += inchem

                for x, y, w, h in birds:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 200, 0), 2)
                frame = putxt(frame, 'Num of birds : ' + str(len(birds)))
                frame = putxt(frame, 'Num of in chem birds : ' + str(tot), (0, 50))

                cv2.imshow('frame', frame, )
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            else:
                self.running = False

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    D = DetectBirds("CHD.mp4")
    D.detect()
