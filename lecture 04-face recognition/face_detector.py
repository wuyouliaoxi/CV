import cv2
from mtcnn import MTCNN
import numpy as np


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=20, tm_threshold=0.7, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

        # ToDo: Specify all parameters for template matching.
        self.tm_threshold = tm_threshold
        self.tm_window_size = tm_window_size

    # ToDo: Track a face in a new image using template matching.
    def track_face(self, image):  # takes an input frame
        # initially detect if required
        if self.reference is None:  # Use the first frame of a video to detect a face using detect_face.
            self.reference = self.detect_face(image)  # Capture the detection as a reference
            return self.reference

        # load template (actually from last frame/reference)
        tem_rect = self.reference["rect"]  # rect：[x, y, width, height]
        template = self.crop_face(self.reference["image"], tem_rect)

        # limit the search window under the assumption of small motions
        extend_top = tem_rect[0] - self.tm_window_size
        extend_left = tem_rect[1] - self.tm_window_size
        extend_width = tem_rect[2] + 2 * self.tm_window_size
        extend_height = tem_rect[3] + 2 * self.tm_window_size
        extend_rect = [extend_top, extend_left, extend_width, extend_height]
        crop_image = self.crop_face(image, extend_rect)

        result = cv2.matchTemplate(crop_image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < self.tm_threshold:  # metric < threshold, means template matching falls
            # re-initialization
            self.reference = self.detect_face(image)
            return self.reference
        else:
            # update reference
            # max_loc is relative coordinate
            face_rect = [max_loc[0] + extend_top, max_loc[1] + extend_left,
                         tem_rect[2], tem_rect[3]]
            face_align = self.align_face(image, face_rect)
            return {"rect": face_rect, "image": image, "aligned": face_align, "response": 0}

    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.
        detections = self.detector.detect_faces(image)  # return the box, confidence, key points
        if not detections:
            self.reference = None
            return None

        # Select face with largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])  # largest bounding box
        face_rect = detections[largest_detection]["box"]  # box：[x, y, width, height] -location of the face

        # Align the detected face.
        aligned = self.align_face(image, face_rect)  # alignment: crop and resize
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

    # Face alignment to predefined size (224,224).
    def align_face(self, image, face_rect):
        return cv2.resize(self.crop_face(image, face_rect), dsize=(self.aligned_image_size, self.aligned_image_size))

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]
