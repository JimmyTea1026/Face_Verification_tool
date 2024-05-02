import os
import numpy as np
import cv2

class Scrfd:
    def __init__(self, det_size, thresh=0.6):
        self.conf_thresh = thresh
        self.det_size = det_size
        self.initParm()

    def initParm(self):
        ##['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32', 'kps_8', 'kps_16', 'kps_32']
        self.nms_thresh = 0.3
        self.num_anchors = 2
        self.fmc = 3
        self.strides =[8,16,32]
        self.max_num = 100
    
    def preprocess(self, image): 
        '''
        input : image(bgr)
        output : det_img(rgb) -> 1*3*640*640
        
        Resize the image to match the input size proportionally,
        normalize the image and put the resized image into the left-top corner of the input size image and fill the rest with zeros.
        '''
        image_ratio = image.shape[0] / image.shape[1]
        input_ratio = self.det_size[1] / self.det_size[0] # h / w

        if image_ratio > input_ratio:
            self.new_height = self.det_size[1]
            self.new_width = int(self.new_height / image_ratio)
        else:
            self.new_width = self.det_size[0]
            self.new_height = int(self.new_width * image_ratio)

        self.det_scale = self.new_height / image.shape[0]

        # resize, convert to RGB and normalize
        resized_img = cv2.resize(image, (self.new_width, self.new_height))
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        np_img = np.array(resized_img).astype(np.float32)
        np_img = (np_img-127.5) / 128
        
        # fill the rest with zeros        
        det_img = np.zeros( (self.det_size[1], self.det_size[0], 3), dtype=np.float32 ) # h,w
        det_img[:self.new_height, :self.new_width, :] = np_img

        img_data = np.transpose(det_img, (2, 0, 1)) 
        img_data = np.expand_dims(img_data, axis=0)

        return img_data
    
    def clipCoord(self, results, img_shape):
        '''
        Ensure the coordinates of the face information are within the image
        '''
        results[:,0:-1:2] = np.clip(results[:,0:-1:2], 0, img_shape[1])    #width
        results[:,1:-1:2] = np.clip(results[:,1:-1:2], 0, img_shape[0])    #height

        return results

    def distance2bbox(self, points, distance):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        
        return np.stack([x1, y1, x2-x1, y2-y1], axis=-1)

    def distance2kps(self, points, distance):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i%2] + distance[:, i]
            py = points[:, i%2+1] + distance[:, i+1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)
    
    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 0] + dets[:, 2]
        y2 = dets[:, 1] + dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
    
    def postprocess(self, outputs, img):
        '''
        Postprocess the output of the model
        The return is a list of face information, each face information is a list of 15 elements
        x, y, w, h, left_eyes_x, left_eyes_y, right_eyes_x, right_eyes_y, 
        nose_x, nose_y, left_mouth_x, left_mouth_y, right_mouth_x, right_mouth_y, confidence 
        '''
        scores_list = []
        bboxes_list = []
        kpss_list = []

        for idx, stride in enumerate(self.strides):
            scores = outputs[idx][0]
            bbox_preds = outputs[idx + self.fmc][0] * stride
            kps_preds = outputs[idx + self.fmc * 2][0] * stride

            h = self.det_size[1] // stride
            w = self.det_size[0] // stride

            ##可寫入 初始化func,
            anchor_centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
            anchor_centers = np.stack([anchor_centers]*self.num_anchors, axis=1).reshape( (-1,2) )

            pos_inds = np.where(scores>=self.conf_thresh)[0]
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            kpss = self.distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))


            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            pos_kpss = kpss[pos_inds]

            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            kpss_list.append(pos_kpss)


        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / self.det_scale
        kpss = np.vstack(kpss_list) / self.det_scale

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        kpss = kpss[order,:,:]

        keep = self.nms(pre_det)

        bboxes = pre_det[keep, :]
        kpss = kpss[keep,:,:]
        if bboxes.shape[0] == 0:
            return None

        kpss = kpss.reshape((-1, 10))

        results = np.concatenate((bboxes[:,:4], kpss, bboxes[:, -1:]), axis=1)
        results = self.clipCoord(results, img.shape)
        
        face_infos = self.get_infos(results)

        return face_infos

    def get_infos(self, det_result):
        face_infos = []
        for result in det_result:
            confidence = result[-1]
            result = result.astype(np.int32)
            face_info = {"x": result[0], "y": result[1], "w": result[2], "h": result[3], "confidence": confidence,
                    "right_eye": (result[4], result[5]), "left_eye": (result[6],result[7]),
                    "nose": (result[8], result[9]), "right_mouth": (result[10], result[11]), "left_mouth": (result[12], result[13])}
            face_infos.append(face_info)
            
        return face_infos
    
    def find_largest_face(self, face_infos):
        '''
        Find the largest face from det_result,
        then align and crop the face from the frame.
        Return the largest face information and the cropped face.
        '''
        face_area = []
        for face_info in face_infos:
            face_area.append(face_info['w'] * face_info['h'])
        largest_face_info = face_infos[np.argmax(face_area)]
        
        return largest_face_info
    
    def align_and_crop(self, image, landmarks):
        """
        對齊人臉並轉換邊界框
        :param image: 原始圖像
        :param landmarks: 人臉五個關鍵點的列表，每個元素是(x, y)座標
        :return: 對齊後的圖像、轉換後的邊界框和擷取出的人臉
        """
        # 確定眼睛的中心點
        eye_left = landmarks['left_eye']
        eye_right = landmarks['right_eye']
        eye_center = ((eye_left[0] + eye_right[0]) // 2, (eye_left[1] + eye_right[1]) // 2)

        # 計算旋轉角度
        dy = eye_right[1] - eye_left[1]
        dx = eye_right[0] - eye_left[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # 計算旋轉中心和旋轉矩陣
        center = (image.shape[1]//2, image.shape[0]//2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 旋轉圖像
        rotated_image = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))

        bbox = [landmarks['x'], landmarks['y'], landmarks['w'], landmarks['h']]
        # 計算新的邊界框座標
        bbox_points = np.array([
            [bbox[0], bbox[1]],
            [bbox[0] + bbox[2], bbox[1]],
            [bbox[0] + bbox[2], bbox[1] + bbox[3]],
            [bbox[0], bbox[1] + bbox[3]]
        ])
        ones = np.ones(shape=(len(bbox_points), 1))
        points_ones = np.hstack([bbox_points, ones])

        # 變換邊界框
        transformed_points = rot_mat.dot(points_ones.T).T
        x_min, y_min = np.min(transformed_points, axis=0)[:2]
        x_max, y_max = np.max(transformed_points, axis=0)[:2]
        transformed_bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

        # 擷取人臉
        x, y, w, h = transformed_bbox
        face_cropped = rotated_image[y:y+h, x:x+w]

        return face_cropped