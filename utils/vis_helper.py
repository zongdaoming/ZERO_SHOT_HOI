from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)

# Standard Library
import os

# Import from third library
import cv2
import matplotlib.pyplot as plt  # noqa E402
import numpy as np
from matplotlib.lines import Line2D  # noqa E402
from matplotlib.patches import Polygon  # noqa E402
from itertools import chain
import pycocotools.mask as cocomask

# Matplotlib requires certain adjustments in some environments
# Must happen before importing matplotlib
# """Set matplotlib up."""
# import matplotlib
# matplotlib.use('Agg')  # Use a non-interactive backend


plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)


def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing code."""
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes


def get_class_string(class_index, score, dataset):
    if dataset is not None and getattr(dataset, 'classes', None) is not None:
        class_text = dataset.classes[class_index]
    else:
        class_text = 'id:{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0').rstrip('.0')


def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""
    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    if show_border:
        # _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)


def vis_class(img, pos, class_str, font_scale=0.35):
    """Visualizes the class."""
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, _GREEN, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    return img


def vis_bbox(img, bbox, thick=1):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), _GREEN, thickness=thick)
    return img


def vis_bad_case_helper(box, label, text, ax, color_list, box_alpha=1):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    color_box = color_list[label % len(color_list), 0:3]
    ax.add_patch(
        plt.Rectangle((box[0], box[1]),
                      box[2] - box[0],
                      box[3] - box[1],
                      fill=True,
                      color=color_box,
                      linewidth=0.5,
                      alpha=box_alpha * 0.5))
    box_w = x2 - x1
    box_h = y2 - y1
    len_ratio = 0.2
    d = min(box_w, box_h) * len_ratio
    corners = list()
    # top left
    corners.append([(x1, y1 + d), (x1, y1), (x1 + d, y1)])
    # top right
    corners.append([(x2 - d, y1), (x2, y1), (x2, y1 + d)])
    # bottom left
    corners.append([(x1, y2 - d), (x1, y2), (x1 + d, y2)])
    # bottom right
    corners.append([(x2 - d, y2), (x2, y2), (x2, y2 - d)])
    line_w = 2 if d * 0.4 > 2 else d * 0.4
    for corner in corners:
        (line_xs, line_ys) = zip(*corner)
        ax.add_line(Line2D(line_xs, line_ys, linewidth=line_w, color=color_box))
        ax.text(x1, y1 - 5, text, fontsize=5,
                family='serif',
                bbox=dict(facecolor=color_box, alpha=0.4, pad=0,
                          edgecolor='none'),
                color='white')
    return ax


def vis_bad_case_helper_test(box, x, y, label, text, ax, color_list, box_alpha=1):    
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    color_box = color_list[label % len(color_list), 0:3]
    ax.add_patch(
        plt.Rectangle((box[0], box[1]),
                      box[2] - box[0],
                      box[3] - box[1],
                      fill=False,
                      color=color_box,
                      linewidth=2.0,
                      alpha=box_alpha * 0.5)
                      )
    ax.plot(x,y,linewidth = 3.0)

    box_w = x2 - x1
    box_h = y2 - y1
    len_ratio = 0.2
    d = min(box_w, box_h) * len_ratio
    corners = list()
    # top left
    corners.append([(x1, y1 + d), (x1, y1), (x1 + d, y1)])
    # top right
    corners.append([(x2 - d, y1), (x2, y1), (x2, y1 + d)])
    # bottom left
    corners.append([(x1, y2 - d), (x1, y2), (x1 + d, y2)])
    # bottom right
    corners.append([(x2 - d, y2), (x2, y2), (x2, y2 - d)])
    line_w = 2 if d * 0.4 > 2 else d * 0.4
    for corner in corners:
        (line_xs, line_ys) = zip(*corner)
        ax.add_line(Line2D(line_xs, line_ys, linewidth=line_w, color=color_box))
        ax.text(x1, y1 - 5, text, fontsize=5,
                family='serif',
                bbox=dict(facecolor=color_box, alpha=0.4, pad=0,
                          edgecolor='none'),
                color='white')
    return ax


def vis_bad_case_with_text(img, instances, missing_list, predict_list, mix_root, class_names, thick=1, dpi=80):

    def get_axes():
        fig, axes = plt.subplots(2, 2)
        # fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(img.shape[1] / dpi * 2, img.shape[0] / dpi * 2)
        # fig.set_size_inches(img.shape[1] / dpi * 2, img.shape[0] / dpi * 2)
        axes_list = list(chain(*axes))
        # axes_list = list(axes)

        title_list = [
            'ground_truth \n{class name}: {class id} ',
            'false negative/missing \n {class name}: {class id} ',
            'false positive/mistake \n {pred class name}: {pred class id}, {score} ',
            'true positive \n {pred class name}: {pred class id}, {score}'
        ]
        for ax, title in zip(axes_list, title_list):
            ax.axis('off')
            ax.set_title(title)
            ax.imshow(img)
        return fig, axes_list

    color_list = colormap(rgb=True) / 255
    fig, axes_list = get_axes()
    # visualize ground truth and false negatives
    for idx, ins in enumerate(instances):
        bbox = ins['bbox']
        label = ins['label']
        name = class_names[label]
        axes_list[0] = vis_bad_case_helper(bbox, label, f'{name}: {label}', axes_list[0], color_list)
        if idx in missing_list:
            axes_list[1] = vis_bad_case_helper(bbox, label, f'{name}: {label}', axes_list[1], color_list)

    # visualize false positives and true positives
    for ins in predict_list:
        bbox = ins['bbox']
        label = ins['label']
        name = class_names[label]
        score = round(ins['score'], 2)
        if not ins['is_right']:
            # false positive
            axes_list[2] = vis_bad_case_helper(bbox, label, f'{name}: {label}, {score}', axes_list[2], color_list)
        else:
            # true positives
            axes_list[3] = vis_bad_case_helper(bbox, label, f'{name}: {label}, {score}', axes_list[3], color_list)
    for ax in axes_list:
        ax.imshow(img)
    return fig



def vis_bad_case_with_text_test(img, instances, hoi_gt, missing_list, predict_list, predict_hoi, mix_root, \
                                     human_class_names, object_class_names, hoi_class_names, thick=1, dpi=500):
    
    def get_axes():
        fig, axes = plt.subplots(2, 2)
        # fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(img.shape[1] / dpi * 2, img.shape[0] / dpi * 2)
        # fig.set_size_inches(img.shape[1] / dpi * 2, img.shape[0] / dpi * 2)
        axes_list = list(chain(*axes))
        # axes_list = list(axes)

        title_list = [
            'ground_truth \n{class name}: {class id} ',
            'false negative/missing \n {class name}: {class id} ',
            'false positive/mistake \n {pred class name}: {pred class id}, {score} ',
            'true positive \n {pred class name}: {pred class id}, {score}'
        ]
        for ax, title in zip(axes_list, title_list):
            ax.axis('off')
            ax.set_title(title)
            ax.imshow(img)
        return fig, axes_list

    color_list = colormap(rgb=True) / 255
    fig, axes_list = get_axes()
    # visualize ground truth and false negatives
    for idx, ins in enumerate(hoi_gt):
        bbox_sub = instances[ins['subject_id']]['bbox']
        label_sub = instances[ins['subject_id']]['category_id']
        bbox_obj = instances[ins['object_id']]['bbox']
        label_obj = instances[ins['object_id']]['category_id']
        label_inter = ins['category_id']
        name_human = human_class_names[label_sub]
        name_object = object_class_names[label_obj]
        name_inter = hoi_class_names[label_inter]
        # pts = (int((bbox_sub[0]+bbox_sub[2])/2), (int(bbox_sub[1]+bbox_sub[3])/2))
        # pto = (int((bbox_obj[0]+bbox_obj[2])/2), (int(bbox_obj[1]+bbox_obj[3])/2))
        x1 = np.linspace(int(bbox_sub[0]+bbox_sub[2])/2, int(bbox_obj[0]+bbox_obj[2])/2, 100)
        y1 = np.linspace(int(bbox_sub[1]+bbox_sub[3])/2, int(bbox_obj[1]+bbox_obj[3])/2, 100)
        axes_list[0] = vis_bad_case_helper_test(bbox_sub, x1, y1, label_sub, f'{name_human}: {label_sub} : {name_inter} ', axes_list[0], color_list)
        axes_list[0] = vis_bad_case_helper_test(bbox_obj, x1, y1, label_obj, f'{name_object}: {label_obj} : {name_inter}', axes_list[0], color_list)
        if idx in missing_list:
            axes_list[1] = vis_bad_case_helper_test(bbox_sub, x1, y1, label_sub, f'{name_human}: {label_sub} : {name_inter}', axes_list[1], color_list)
            axes_list[1] = vis_bad_case_helper_test(bbox_obj, x1, y1, label_obj, f'{name_object}: {label_obj}: {name_inter}', axes_list[1], color_list)

    # visualize false positives and true positives
    for ins in predict_hoi:
        bbox_sub = predict_list[ins['subject_id']]['bbox']
        label_sub = predict_list[ins['subject_id']]['category_id']
        bbox_obj = predict_list[ins['object_id']]['bbox']
        label_obj = predict_list[ins['object_id']]['category_id']
        label_inter = ins['category_id']
        name_human = human_class_names[label_sub]
        name_object = object_class_names[label_obj]
        name_inter = hoi_class_names[label_inter]
        # score = round(ins['score'], 2)
        score = ins['score']
        # name = class_names[label]
        # score = round(ins['score'], 2)
        x1 = np.linspace(int(bbox_sub[0]+bbox_sub[2])/2, int(bbox_obj[0]+bbox_obj[2])/2, 100)
        y1 = np.linspace(int(bbox_sub[1]+bbox_sub[3])/2, int(bbox_obj[1]+bbox_obj[3])/2, 100)
        if not ins.get('is_right', None):
            # false positive
            axes_list[2] = vis_bad_case_helper_test(bbox_sub, x1, y1, label_sub, f'{name_human}: {label_sub} : {name_inter}, {score}', axes_list[2], color_list)
            axes_list[2] = vis_bad_case_helper_test(bbox_obj, x1, y1, label_obj, f'{name_object}: {label_obj} : {name_inter}, {score}', axes_list[2], color_list)
        else:
            # true positives
            axes_list[3] = vis_bad_case_helper_test(bbox_sub, x1, y1, label_sub, f'{name_human}: {label_sub} : {name_inter}, {score}', axes_list[3], color_list)
            axes_list[3] = vis_bad_case_helper_test(bbox_obj, x1, y1, label_obj, f'{name_object}: {label_obj} : {name_inter}, {score}', axes_list[3], color_list)
    for ax in axes_list:
        ax.imshow(img)
    return fig


def vis_keypoints(img, kps, dataset=None, kp_thresh=2, alpha=0.7):
    """
    Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    # dataset_keypoints, _ = keypoint_utils.get_keypoints()
    # kp_lines = kp_connections(dataset_keypoints)
    if dataset is not None:
        kp_lines = dataset.get_keyp_lines()
    else:
        kp_lines = []
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    # mid_shoulder = (
    #     kps[:2, dataset_keypoints.index('right_shoulder')] +
    #     kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    # sc_mid_shoulder = np.minimum(
    #     kps[2, dataset_keypoints.index('right_shoulder')],
    #     kps[2, dataset_keypoints.index('left_shoulder')])
    # mid_hip = (
    #     kps[:2, dataset_keypoints.index('right_hip')] +
    #     kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    # sc_mid_hip = np.minimum(
    #     kps[2, dataset_keypoints.index('right_hip')],
    #     kps[2, dataset_keypoints.index('left_hip')])
    # nose_idx = dataset_keypoints.index('nose')
    # if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
    #     cv2.line(
    #         kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
    #         color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    # if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
    #     cv2.line(
    #         kp_mask, tuple(mid_shoulder), tuple(mid_hip),
    #         color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(kp_mask, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(kp_mask, p1, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(kp_mask, p2, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    if len(kp_lines) == 0:
        for ix in range(kps.shape[1]):
            if kps[2, ix] > kp_thresh:
                p = kps[0, ix], kps[1, ix]
                cv2.circle(kp_mask, p, radius=3, color=colors[ix], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_one_image_opencv(im,
                         boxes,
                         classes,
                         segms=None,
                         keypoints=None,
                         thresh=0.9,
                         kp_thresh=2,
                         show_box=False,
                         dataset=None,
                         show_class=False):
    """Constructs a numpy array with the detections visualized."""

    # if isinstance(boxes, list):
    #     boxes, segms, keypoints, classes = convert_from_cls_format(
    #         boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return im

    if segms is not None and len(segms) > 0:
        # masks = mask_util.decode(segms)
        masks = segms
        color_list = colormap()
        mask_color_id = 0

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue

        # show box (off by default)
        if show_box:
            im = vis_bbox(im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))

        # show class (off by default)
        if show_class:
            class_str = get_class_string(classes[i], score, dataset)
            im = vis_class(im, (bbox[0], bbox[1] - 2), class_str)

        # show mask
        if segms is not None and len(segms) > i:
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1
            im = vis_mask(im, masks[..., i], color_mask)

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            im = vis_keypoints(im, keypoints[i], dataset=dataset, kp_thresh=kp_thresh)

    return im


def vis_one_image(im,
                  im_name,
                  output_dir,
                  boxes,
                  classes,
                  ig_boxes=None,
                  segms=None,
                  keypoints=None,
                  thresh=0.9,
                  kp_thresh=2,
                  dpi=200,
                  box_alpha=0.0,
                  dataset=None,
                  show_class=False,
                  ext='pdf',
                  out_when_no_box=False):
    """Visual debugging of detections."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # if isinstance(boxes, list):
    #     boxes, segms, keypoints, classes = convert_from_cls_format(
    #         boxes, segms, keypoints)

    if (boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh) and not out_when_no_box:
        return

    color_list = colormap(rgb=True) / 255
    cmap = plt.get_cmap('rainbow')

    if keypoints is not None:
        # dataset_keypoints = dataset.get_keypoints()
        kp_lines = dataset.get_keyp_lines()
        colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    if segms is not None and len(segms) > 0:
        # masks = mask_util.decode(segms)
        masks = segms

    # color_list = colormap(rgb=True) / 255

    # kp_lines = kp_connections(dataset_keypoints)
    # cmap = plt.get_cmap('rainbow')
    # colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    if boxes is None:
        sorted_inds = []  # avoid crash when 'boxes' is None
    else:
        # Display in largest to smallest order to reduce occlusion
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)

    if ig_boxes is not None:
        for bbox in ig_boxes:
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1],
                              fill=False,
                              edgecolor='y',
                              linewidth=0.5,
                              alpha=box_alpha))

    mask_color_id = 0
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue
        # show box (off by default)
        if segms is not None:
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1],
                              fill=False,
                              edgecolor='g',
                              linewidth=0.5,
                              alpha=box_alpha))
        else:
            color_box = color_list[classes[i] % len(color_list), 0:3]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1],
                              fill=True,
                              edgecolor='None',
                              color=color_box,
                              linewidth=0.5,
                              alpha=box_alpha * 0.5))
            # draw corners
            x1, y1, x2, y2 = bbox[:4]
            len_ratio = 0.2
            box_w = x2 - x1
            box_h = y2 - y1
            d = min(box_w, box_h) * len_ratio
            corners = list()
            # top left
            corners.append([(x1, y1 + d), (x1, y1), (x1 + d, y1)])
            # top right
            corners.append([(x2 - d, y1), (x2, y1), (x2, y1 + d)])
            # bottom left
            corners.append([(x1, y2 - d), (x1, y2), (x1 + d, y2)])
            # bottom right
            corners.append([(x2 - d, y2), (x2, y2), (x2, y2 - d)])
            line_w = 2 if d * 0.4 > 2 else d * 0.4
            for corner in corners:
                (line_xs, line_ys) = zip(*corner)
                ax.add_line(Line2D(line_xs, line_ys, linewidth=line_w, color=color_box))

        if show_class:
            if segms is not None:
                ax.text(bbox[0],
                        bbox[1] - 2,
                        get_class_string(classes[i], score, dataset),
                        fontsize=3,
                        family='serif',
                        bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                        color='white')
            else:
                ax.text(bbox[0],
                        bbox[1] - 5,
                        get_class_string(classes[i], score, dataset),
                        fontsize=3,
                        family='serif',
                        bbox=dict(facecolor=color_box, alpha=0.4, pad=0, edgecolor='none'),
                        color='white')

        # show mask
        if segms is not None and len(segms) > i:
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = masks[:, :, i]

            contour, hier = cv2.findContours(e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(c.reshape((-1, 2)),
                                  fill=True,
                                  facecolor=color_mask,
                                  edgecolor='w',
                                  linewidth=1.2,
                                  alpha=0.5)
                ax.add_patch(polygon)

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            kps = keypoints[i]
            plt.autoscale(False)
            for l in range(len(kp_lines)):
                i1 = kp_lines[l][0]
                i2 = kp_lines[l][1]
                if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                    x = [kps[0, i1], kps[0, i2]]
                    y = [kps[1, i1], kps[1, i2]]
                    line = plt.plot(x, y)
                    plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
                if kps[2, i1] > kp_thresh:
                    plt.plot(kps[0, i1], kps[1, i1], '.', color=colors[l], markersize=3.0, alpha=0.7)

                if kps[2, i2] > kp_thresh:
                    plt.plot(kps[0, i2], kps[1, i2], '.', color=colors[l], markersize=3.0, alpha=0.7)
            if len(kp_lines) == 0:
                for ix in range(kps.shape[1]):
                    if kps[2, ix] > kp_thresh:
                        plt.plot(kps[0, ix], kps[1, ix], '.', color=colors[ix], markersize=3.0, alpha=0.7)

            # add mid shoulder / mid hip for better visualization
            # mid_shoulder = (
            #     kps[:2, dataset_keypoints.index('right_shoulder')] +
            #     kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            # sc_mid_shoulder = np.minimum(
            #     kps[2, dataset_keypoints.index('right_shoulder')],
            #     kps[2, dataset_keypoints.index('left_shoulder')])
            # mid_hip = (
            #     kps[:2, dataset_keypoints.index('right_hip')] +
            #     kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            # sc_mid_hip = np.minimum(
            #     kps[2, dataset_keypoints.index('right_hip')],
            #     kps[2, dataset_keypoints.index('left_hip')])
            # if (sc_mid_shoulder > kp_thresh and
            #         kps[2, dataset_keypoints.index('nose')] > kp_thresh):
            #     x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
            #     y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
            #     line = plt.plot(x, y)
            #     plt.setp(
            #         line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
            # if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
            #     x = [mid_shoulder[0], mid_hip[0]]
            #     y = [mid_shoulder[1], mid_hip[1]]
            #     line = plt.plot(x, y)
            #     plt.setp(
            #         line, color=colors[len(kp_lines) + 1], linewidth=1.0,
            #         alpha=0.7)

    output_name = os.path.basename(im_name) + '.' + ext
    fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
    plt.close('all')


def colormap(rgb=False):
    color_list = np.array([
        0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556, 0.466, 0.674, 0.188, 0.301,
        0.745, 0.933, 0.635, 0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000, 1.000, 0.500,
        0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000,
        0.333, 0.667, 0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000, 1.000,
        0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
        0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500, 0.333, 1.000, 0.500, 0.667, 0.000, 0.500,
        0.667, 0.333, 0.500, 0.667, 0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333, 0.500, 1.000,
        0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000, 0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000,
        1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000,
        0.667, 0.667, 1.000, 0.667, 1.000, 1.000, 1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167,
        0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000,
        0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
        0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000,
        0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429, 0.429,
        0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714, 0.857, 0.857, 0.857, 1.000, 1.000, 1.000
    ]).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


def cocoseg_to_binary(seg, height, width):
    rle = cocomask.frPyObjects(seg, height, width)
    rle = cocomask.merge(rle)
    mask = cocomask.decode([rle])
    assert mask.shape[2] == 1
    return mask[:, :, 0]


def vis_transform(func):
    def visualize(data, filename):
        img = data['image']
        height, width = img.shape[:2]
        bboxes = data.get('gt_bboxes', None)
        masks = data.get('gt_masks', None)
        mask_2d = None
        if bboxes is not None:
            bboxes = bboxes.detach().numpy()
            classes = bboxes[:, 4].copy()
            bboxes[:, 4] = 1
        if masks is not None:
            mask_2d = []
            for mask in masks:
                _mask = cocoseg_to_binary(mask, height, width)
                mask_2d.append(_mask)
            mask_2d = np.stack(mask_2d, axis=-1)
        img = vis_one_image_opencv(np.ascontiguousarray(img),
                                   bboxes,
                                   classes,
                                   mask_2d,
                                   show_box=bboxes is not None,
                                   thresh=0)
        save_dir = 'vis_augmentation'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, img)
        print(f'save {filename} to {save_dir}')

    def wrapper(*args, **kwargs):
        aug = list(args)[0]
        aug_name = aug.__class__.__name__
        data = list(args)[1]
        filename = data['filename']
        name, postfix = filename.split('.')
        visualize(data, name + '_' + aug_name + '_' + 'before.' + postfix)
        data = func(*args, **kwargs)
        visualize(data, name + '_' + aug_name + '_' + 'after.' + postfix)
        return data

    return wrapper
