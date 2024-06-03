from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from filter import PointTrackerKalmanFilter
import numpy as np

def test_sim(curr_img, lost_img):

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, preprocess = clip.load("ViT-B/32", device=device)

  cos_list = []
  cos = torch.nn.CosineSimilarity(dim=0)

  image1_preprocess = preprocess(Image.fromarray(curr_img)).unsqueeze(0).to(device)
  image1_features = model.encode_image(image1_preprocess)


  image2_preprocess = preprocess(Image.fromarray(lost_img[1])).unsqueeze(0).to(device)
  image2_features = model.encode_image(image2_preprocess)

  similarity = cos(image1_features[0],image2_features[0]).item()
  similarity = (similarity+1)/2

  return similarity

def mse2(img1, img2):
   h, w, _ = img1.shape
   img2 = cv2.resize(img2, (w, h))
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   #print(mse)
   return mse

def lightglue_matching(bbox, object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
    # Initialize the LightGlue model
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features="superpoint").eval().to(device)

    bbox = torch.from_numpy(bbox)
    object = torch.from_numpy(object)

    bbox = bbox.permute(2, 0, 1).float()
    object = object.permute(2, 0, 1).float()

    feats0 = extractor.extract(bbox.to(device))
    feats1 = extractor.extract(object.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    return len(matches)

def reid2(frame, detections, id, bbox):
    index = 0
    best_sim = [0, 0.0]
    if id in detections.tracker_id:
        for index in range(len(detections.tracker_id)):
            if detections.tracker_id[index] == id:
                break
        if bbox.shape[0] * bbox.shape[1] < detections.xyxy[index][2] * detections.xyxy[index][3]:
            bbox = frame[int(detections.xyxy[index][0]):int(detections.xyxy[index][2]), int(detections.xyxy[index][1]):int(detections.xyxy[index][3])]
    else:
        for index in range(len(detections.tracker_id)):
            curr = frame[int(detections.xyxy[index][0]):int(detections.xyxy[index][2]), int(detections.xyxy[index][1]):int(detections.xyxy[index][3])]
            sim = lightglue_matching(bbox, curr)
            if sim > best_sim[1]:
                print("---------------------------------------")
                print(sim)
                print(index)
                best_sim[0] = index
                best_sim[1] = sim
        bbox = frame[int(detections.xyxy[best_sim[0]][0]):int(detections.xyxy[best_sim[0]][2]), int(detections.xyxy[best_sim[0]][1]):int(detections.xyxy[best_sim[0]][3])]
        detections.tracker_id[best_sim[0]] = id
    return detections, bbox

def get_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    boxA : list or tuple
        Bounding box A in the format [x1, y1, x2, y2]
    boxB : list or tuple
        Bounding box B in the format [x1, y1, x2, y2]

    Returns:
    float
        The IoU of boxA and boxB
    """

    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def reid(detections, id, point):
    reid = 0
    best = 0
    if id in detections.tracker_id:
        return id
    else:
        target_bbox = [point[0] - 20, point[1] - 30, point[0] + 20, point[1]]
        for index in range(len(detections.tracker_id)):
            bbox = detections.xyxy[index]
            iou = get_iou(target_bbox, bbox)
            if iou > best:
                reid = detections.tracker_id[index]
        return reid

def get_reid_dict(dict, kalman, detections):
    best = 0
    reid = kalman[1]
    point = kalman[0].get_predicted_state()

    target_bbox = [point[0] - 20, point[1] - 30, point[0] + 20, point[1] +30]

    for index in range(len(detections.tracker_id)):
        bbox = detections.xyxy[index]
        iou = get_iou(target_bbox, bbox)
        if iou > best:
            best = iou
            reid = detections.tracker_id[index]

    dict[str(kalman[1])] = reid
    kalman[3] = best
    return dict, kalman

def reid_all(kalmans, last_id, detections, dict):
    # Kalman filter attributes
    initial_covariance = np.eye(2) * 1000  # Initial covariance estimate
    process_noise = np.eye(2) * 0.01  # Process noise covariance
    measurement_noise = np.eye(2) * 0.1  # Measurement noise covariance

    for kalman in kalmans:
        kalman[0].predict()
        dict, kalman = get_reid_dict(dict, kalman, detections)


    for kalman in kalmans:
        if kalman[3] == 0.0:
            kalman[2] = kalman[2] + 1
            kalman[0].update(kalman[0].get_predicted_state())
            if kalman[2] > 30:
                kalmans.remove(kalman)

        else:
            kalman[2] = 0
            for i in range(len(detections.tracker_id)):
                if detections.tracker_id[i] in dict.values():
                    if dict[str(kalman[1])] == detections.tracker_id[i]:
                        detections.tracker_id[i] = kalman[1]
                        dict[str(kalman[1])] = kalman[1]
                        kalman[0].update([(detections.xyxy[i][0] + detections.xyxy[i][2]) / 2,
                                  (detections.xyxy[i][1] + detections.xyxy[i][3]) / 2])
                    else:
                        pass

    for i in range(len(detections.tracker_id)):
        if detections.tracker_id[i] not in dict.values():
            last_id = last_id + 1
            initial_state = np.array([(detections.xyxy[i][0] + detections.xyxy[i][2]) / 2,
                                    (detections.xyxy[i][1] + detections.xyxy[i][3]) / 2])
            kalmans.append([PointTrackerKalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise), last_id, 0,
                             0.0])

    return kalmans, last_id, detections, dict