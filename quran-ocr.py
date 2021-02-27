from os import path
import cv2
import csv
import math
import numpy as np
import statistics as stat
import xml.etree.ElementTree as etree


# TODO: make all constants as parameters
TEMPLATE_MATCHING_THRESHOLD = 0.5
NMS_THRESHOLD = 0.2

# TODO: define page border automatically
PAGE_START = 3
PAGE_END = 604
LINE_NUMBERS = 15  # only estimation, since there are some special cases
PAGE_BORDER = 42  # only estimation for page 3 and above
AYAH_MARGIN_TOLERANCE = PAGE_BORDER + 30

IMAGE_INPUT_DIR = './dataset/quran-images/'
IMAGE_OUTPUT_DIR = './dataset/quran-images/output/'  # for validation only
IMAGE_FORMAT = 'png'

SURAH_MARKER_FILENAME = 'sura-name-marker'
BISMILLAH_MARKER_FILENAME = 'bismilla-marker'
AYAH_MARKER_FILENAME = 'aya-end-marker'

_ = cv2.imread(path.join(IMAGE_INPUT_DIR, f'1.{IMAGE_FORMAT}'))
PAGE_HEIGHT, PAGE_WIDTH, _ = _.shape  # TODO: should be resized

METADATA_INPUT_FILEPATH = './dataset/quran-metadata.xml'
METADATA_OUTPUT_FILEPATH = './dataset/quran-index.csv'

GENERATE_INDEXING = True
GENERATE_PREVIEWS = True


class Template:

    def __init__(
        self,
        img_path,
        label,
        color,
        matching_threshold=TEMPLATE_MATCHING_THRESHOLD
    ):
        self.img_path = img_path
        self.label = label
        self.color = color
        self.template = cv2.imread(img_path)
        self.template_height, self.template_width = self.template.shape[:2]
        self.matching_threshold = matching_threshold


def compute_iou(
    a,
    b,
    epsilon=1e-5
):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    width = (x2 - x1)
    height = (y2 - y1)

    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    return area_overlap / (area_combined + epsilon)


def non_max_suppression(
    objects,
    thres=0.5,
    score_key='match_value'
):
    def bbox(object_):
        return [
            object_['top_left_x'],
            object_['top_left_y'],
            object_['bottom_right_x'],
            object_['bottom_right_y'],
        ]

    sorted_objects = sorted(objects, key=lambda obj: obj[score_key],
                            reverse=True)
    filtered_objects = []
    for object_ in sorted_objects:
        overlap_found = False
        for filtered_object in filtered_objects:
            iou = compute_iou(bbox(object_), bbox(filtered_object))
            if iou > thres:
                overlap_found = True
                break
        if not overlap_found:
            filtered_objects.append(object_)
    return filtered_objects


def find_match(
    img,
    templates
):
    detections = []
    for template in templates:
        template_matching = cv2.matchTemplate(
            template.template, img, cv2.TM_CCOEFF_NORMED
        )

        match_locations = np.where(template_matching >= template.matching_threshold)

        for (x, y) in zip(match_locations[1], match_locations[0]):
            match = {
                'top_left_x': x,
                'top_left_y': y,
                'bottom_right_x': x + template.template_width,
                'bottom_right_y': y + template.template_height,
                'match_value': template_matching[y, x],
                'label': template.label,
                'color': template.color
            }
            detections.append(match)

    return non_max_suppression(
        detections, thres=NMS_THRESHOLD
    )


# TODO: resize all imgs for production
# TODO: optimise the code
ref_bboxes = []
ref_idx_sura = 2 # TODO: must be `PAGE_START`
ref_idx_aya = 6 # TODO: must be `1`
ref_suras = etree.parse(METADATA_INPUT_FILEPATH).find('suras').findall('sura')

for page_no in range(PAGE_START, PAGE_END + 1):
    img = cv2.imread(path.join(IMAGE_INPUT_DIR, f'{page_no}.{IMAGE_FORMAT}'))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # This is a naive method
    # to remove page border
    # for page 3-604 only
    img_borderless = 255 * np.ones((PAGE_HEIGHT, PAGE_WIDTH), np.uint8)
    img_borderless[
        PAGE_BORDER:(PAGE_HEIGHT-PAGE_BORDER),
        PAGE_BORDER:(PAGE_WIDTH-PAGE_BORDER)
    ] = img_gray[
        PAGE_BORDER:(PAGE_HEIGHT-PAGE_BORDER),
        PAGE_BORDER:(PAGE_WIDTH-PAGE_BORDER)
    ]

    # Find surah, bismillah, and ayah's end marker glyphs
    templates = [
        Template(
            img_path=path.join(IMAGE_INPUT_DIR,
            f'{SURAH_MARKER_FILENAME}.{IMAGE_FORMAT}'),
            label='surah',
            color=(255, 0, 0)
        ),
        Template(
            img_path=path.join(IMAGE_INPUT_DIR,
            f'{AYAH_MARKER_FILENAME}.{IMAGE_FORMAT}'),
            label='ayah',
            color=(0, 0, 255)
        ),
        Template(
            img_path=path.join(IMAGE_INPUT_DIR,
            f'{BISMILLAH_MARKER_FILENAME}.{IMAGE_FORMAT}'),
            label='bismillah',
            color=(0, 255, 0)
        ),
    ]
    bbox_decors = find_match(img, templates)

    # Remove surah and bismillah glyphs
    for bbox in bbox_decors:
        if bbox['label'] == 'ayah':
            continue
        img_borderless[
            bbox['top_left_y']:bbox['bottom_right_y'],
            bbox['top_left_x']:bbox['bottom_right_x']
        ] = 255

    # Line segmentation
    # TODO: need improvements
    _, img_thresh = cv2.threshold(
        img_borderless, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    bbox_lines = cv2.findContours(
        img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bbox_lines = bbox_lines[0] if len(bbox_lines) == 2 else bbox_lines[1]

    nodes = []
    heights = []
    for bbox in bbox_lines:
        x, y, w, h = cv2.boundingRect(bbox)
        # TODO: do not hard code this
        if w < 15 and h < 12:  # remove all possible diacritics (small glyphs)
            continue
        nodes.append({
            'top_left_x': x,
            'top_left_y': y,
            'bottom_right_x': x + w,
            'bottom_right_y': y + h,
            'head': (x, y + h // 2),
            'tail': (x + w, y + h // 2),
            'is_visited': False
        })
        heights.append(h)

    nodes.sort(key=lambda x: x['tail'][0])
    len_nodes = len(nodes)
    # FIXME: using median heights isn't cannot be accounted for
    # in knowing whether two nodes are on separate lines
    med_heights = stat.median(heights)

    transports = [[0 for _ in range(len_nodes)] for _ in range(len_nodes)]
    for idx_source in range(len_nodes):
        pt_source = nodes[idx_source]
        for idx_target in range(idx_source + 1, len_nodes):
            pt_target = nodes[idx_target]
            transports[idx_source][idx_target] = \
                math.dist(pt_target['head'], pt_source['tail'])
            transports[idx_target][idx_source] = \
                math.dist(pt_source['head'], pt_target['tail'])

    lines = []  # will be list of node index list
    for idx_source in range(len_nodes):
        if nodes[idx_source]['is_visited']:
            continue
        lines.append([idx_source])
        nodes[idx_source]['is_visited'] = True

        idx_source_ = idx_source
        target_idxs = [
            idx_target for idx_target in range(len_nodes)
            if not nodes[idx_target]['is_visited'] and
            abs(nodes[idx_target]['head'][1] -
                nodes[idx_source]['tail'][1])
            < med_heights
        ]
        while target_idxs:
            idx_next_node = target_idxs[
                np.argmin([transports[idx_source_][idx_target]
                           for idx_target in target_idxs])
            ]
            lines[-1].append(idx_next_node)
            target_idxs.remove(idx_next_node)
            nodes[idx_next_node]['is_visited'] = True

            idx_source_ = idx_next_node
            target_idxs = [
                idx_target for idx_target in range(len_nodes)
                if not nodes[idx_target]['is_visited'] and
                abs(nodes[idx_target]['head'][1] -
                    nodes[idx_source]['tail'][1])
                < med_heights
            ]

    # TODO: reassign all diacritics
    lines = [line for line in lines if len(line) >= 5]  # to handle if there are
                                                        # still diacritics

    # Build line bboxes
    bbox_lines_ = []
    for line in lines:
        nodes_ = [nodes[idx_node] for idx_node in line]
        match = {
            'top_left_x': min([node['top_left_x'] for node in nodes_]),
            'top_left_y': min([node['top_left_y'] for node in nodes_]),
            'bottom_right_x': max([node['bottom_right_x'] for node in nodes_]),
            'bottom_right_y': max([node['bottom_right_y'] for node in nodes_]),
            'label': 'line',
            'color': (0, 255, 255)
        }

        is_overlaped = False
        for bbox in bbox_lines_:
            iou = compute_iou([
                match['top_left_x'], match['top_left_y'],
                match['bottom_right_x'], match['bottom_right_y']
            ], [
                bbox['top_left_x'], bbox['top_left_y'],
                bbox['bottom_right_x'], bbox['bottom_right_y']
            ])
            if iou > 0.25:  # 0.25 for NMS threshold
                bbox['top_left_x'] = \
                    min([match['top_left_x'], bbox['top_left_x']])
                bbox['top_left_y'] = \
                    min([match['top_left_y'], bbox['top_left_y']])
                bbox['bottom_right_x'] = \
                    max([match['bottom_right_x'], bbox['bottom_right_x']])
                bbox['bottom_right_y'] = \
                    max([match['bottom_right_y'], bbox['bottom_right_y']])
                is_overlaped = True
                break

        if not is_overlaped:
            bbox_lines_.append(match)

    # Gather all bboxes
    bbox_all_lines = [bbox for bbox in bbox_decors if bbox['label'] != 'ayah']
    bbox_all_lines += bbox_lines_
    bbox_all_lines.sort(key=lambda x: x['top_left_y'])

    if len(bbox_all_lines) != LINE_NUMBERS:
        print(f'Warning! There is {len(bbox_all_lines)} lines detected on '
              f'page {page_no}')

    bbox_ayas = [bbox for bbox in bbox_decors if bbox['label'] == 'ayah']
    if len(bbox_ayas) == 0:
        continue
    bbox_ayas.sort(key=lambda x: x['top_left_y'])

    # Normalise
    # to remove gap between bbox lines
    bbox_all_lines_norm = bbox_all_lines.copy()
    for idx_line, bbox in enumerate(bbox_all_lines_norm):
        try:
            y_center = (
                bbox['bottom_right_y'] + \
                bbox_all_lines_norm[idx_line + 1]['top_left_y']
            ) // 2
            bbox_all_lines_norm[idx_line]['bottom_right_y'] = y_center
            bbox_all_lines_norm[idx_line + 1]['top_left_y'] = y_center
        except:
            break

    # to uniform bbox line lengths
    for idx, _ in enumerate(bbox_all_lines_norm):
        bbox_all_lines_norm[idx]['top_left_x'] = PAGE_BORDER
        bbox_all_lines_norm[idx]['bottom_right_x'] = PAGE_WIDTH - PAGE_BORDER

    # TODO: uniform line height for all `bbox[label=line]` only
    bbox_all_lines_norm[0]["TOP_LEFT_Y"] = \
        min(PAGE_BORDER, bbox_all_lines_norm[0]["TOP_LEFT_Y"])
    bbox_all_lines_norm[-1]["BOTTOM_RIGHT_Y"] = max(
        PAGE_HEIGHT - PAGE_BORDER, bbox_all_lines_norm[-1]["BOTTOM_RIGHT_Y"]
    )

    # Ayah segmentation
    bboxes = []
    idx_aya = 0
    for idx_line, bbox_line in enumerate(bbox_all_lines_norm):
        if bbox_line['label'] != 'line':
            bboxes.append(bbox_line)
            continue

        def get_y_center(bbox):
            return bbox['top_left_y'] + (
                bbox['bottom_right_y'] - bbox['top_left_y']
            ) / 2

        bbox_ayas_ = []
        y_center = get_y_center(bbox_ayas[idx_aya])
        while bbox_line['top_left_y'] < y_center < bbox_line['bottom_right_y']:
            bbox_ayas_.append(bbox_ayas[idx_aya])
            idx_aya += 1
            try:
                y_center = get_y_center(bbox_ayas[idx_aya])
            except:
                break

        if len(bbox_ayas_) > 0:
            bbox_ayas_.sort(key=lambda x: x['top_left_x'], reverse=True)
            margin_right = bbox_line['bottom_right_x']
            for idx_aya_, bbox_aya in enumerate(bbox_ayas_):
                bbox_line_ = bbox_line.copy()
                bbox_line_['top_left_x'] = bbox_aya['top_left_x']
                try:
                    if bbox_aya['top_left_x'] < AYAH_MARGIN_TOLERANCE or (
                        (idx_aya_ + 1) == len(bbox_ayas_) and
                        bbox_all_lines_norm[idx_line + 1]['label'] != 'line'
                    ):
                        bbox_line_['top_left_x'] = bbox_line['top_left_x']
                except:
                    bbox_line_['top_left_x'] = bbox_line['top_left_x']
                bbox_line_['bottom_right_x'] = margin_right
                bbox_line_['label'] = bbox_aya['label']
                bboxes.append(bbox_line_)
                margin_right = bbox_line_['top_left_x']

            if bbox_line_['top_left_x'] > PAGE_BORDER:
                bbox_line__ = bbox_line.copy()
                bbox_line__['bottom_right_x'] = bbox_line_['top_left_x']
                bboxes.append(bbox_line__)
        else:
            bboxes.append(bbox_line)

    if not GENERATE_INDEXING and not GENERATE_PREVIEWS:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_aya_segmented = img_rgb.copy()
    for idx, bbox in enumerate(bboxes):
        if GENERATE_PREVIEWS:
            cv2.rectangle(
                img_aya_segmented,
                (bbox['top_left_x'], bbox['top_left_y']),
                (bbox['bottom_right_x'], bbox['bottom_right_y']),
                bbox['color'],
                1,
            )
            cv2.putText(
                img_aya_segmented,
                f"{idx + 1} {bbox['label']}",
                (bbox['top_left_x'] + 2, bbox['top_left_y'] + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                bbox['color'],
                1,
                cv2.LINE_AA,
            )

        if GENERATE_INDEXING:
            if bbox['label'] in ['surah', 'bismillah']:
                aya_no = -1
            else:
                aya_no = ref_idx_aya

            ref_bboxes.append({
                'page': page_no,
                'sura': ref_suras[ref_idx_sura - 1].get('index'),
                'aya': aya_no,
                'role': 'ayah' if bbox['label'] == 'line' else bbox['label'],
                'x1': bbox['top_left_x'],
                'y1': bbox['top_left_y'],
                'x2': bbox['bottom_right_x'],
                'y2': bbox['bottom_right_y'],
            })

            if bbox['label'] == 'surah':
                ref_idx_sura += 1
                ref_idx_aya = 1
            elif bbox['label'] == 'ayah':
                ref_idx_aya += 1

    if GENERATE_PREVIEWS:
        img_bgr = cv2.cvtColor(img_aya_segmented, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            path.join(IMAGE_OUTPUT_DIR, f'{page_no}.{IMAGE_FORMAT}'), img_bgr
        )

    print(f'Page {page_no} successfully extracted')

if GENERATE_INDEXING:
    keys = ref_bboxes[0].keys()
    with open(METADATA_OUTPUT_FILEPATH, 'w+', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(ref_bboxes)
