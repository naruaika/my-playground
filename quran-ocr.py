from os import listdir, makedirs, path
from pdf2image import convert_from_path
import csv
import cv2
import numpy as np
import sys
import tempfile
import xml.etree.ElementTree as etree


QURAN_TYPES = {  # from Qurancomplex.gov.sa
    'hafs-standard39': {
        'page-start': 3,  # page where Al-Faatihah is located (index from zero)
        'page-end': 607,  # page where An-Naas is located
        'dpi': 190
    },
    'hafs-jawamee39': {
        'page-start': 3,
        'page-end': 607,
        'dpi': 190
    },

    # for me finding a segmentation algorithm for each printing
    'sample': {
        'page-start': 3,
        'page-end': 8,
        'dpi': 190
    },
    'sample2': {
        'page-start': 3,
        'page-end': 8,
        'dpi': 190
    }
}


# User-defined variables
QURAN_TYPE = 'hafs-jawamee39'
PAGE_OUTPUT_WIDTH = 540  # in pixel

QURAN_INPUT_FILEPATH = '/home/naru/Repositories/my-playground/dataset/' \
                       f'quran-images/{QURAN_TYPE}.pdf'
QURAN_OUTPUT_DIR = '/home/naru/Repositories/my-playground/dataset/' \
                   'quran-images/output/'

DIACRITICS_DIR = '/home/naru/Repositories/my-playground/dataset/quran-glyphs/'
IMAGE_FORMAT = 'png'

METADATA_INPUT_FILEPATH = '/home/naru/Repositories/my-playground/dataset/' \
                          'quran-metadata.xml'  # from Tanzil.net
METADATA_OUTPUT_FILEPATH = path.join(QURAN_OUTPUT_DIR,
                                     f'{QURAN_TYPE}-bboxes.csv')


# Application variables
PAGE_START = QURAN_TYPES[QURAN_TYPE]['page-start']
PAGE_END = QURAN_TYPES[QURAN_TYPE]['page-end']
PAGE_DPI = QURAN_TYPES[QURAN_TYPE]['dpi']

TEMPLATE_MATCHING_THRESHOLD = 0.55
NMS_THRESHOLD = 0.25

SURA_START = 1  # default is `1`
AYAH_START = 1  # default is `1`
LINE_NUMBERS = (7, 15)  # for the first two pages and the default pages
SPECIAL_12PAGES = True  # if the first two pages have a special design;
                        # as they usually have smaller font size

GENERATE_INDEXING = True
GENERATE_PREVIEWS = True
VERBOSE_MODE = True


print('Bismillaah.')

print(f'{" " * 8} Reading input Quran file ...')
with tempfile.TemporaryDirectory() as outdir:
    try:
        PAGES = convert_from_path(QURAN_INPUT_FILEPATH, dpi=PAGE_DPI,
                                  output_folder=outdir)
        # PAGES = convert_from_path(QURAN_INPUT_FILEPATH, dpi=PAGE_DPI)
    except:
        print('[Failed] Input Quran file is not found.')
        sys.exit()
PAGE_WIDTH, PAGE_HEIGHT = PAGES[0].size
PAGE_SCALE = PAGE_OUTPUT_WIDTH / PAGE_WIDTH
print('[  Ok  ] Read input Quran file.')

print(f'{" " * 8} Checking output directory ...')
if not path.isdir(QURAN_OUTPUT_DIR):
    print(f'{" " * 8} Output directory is not found. Creating it ...')
    makedirs(QURAN_OUTPUT_DIR, exist_ok=True)
print('[  Ok  ] Checked output directory.')

print(f'{" " * 8} Checking diacritic directory ...')
if not path.isdir(DIACRITICS_DIR):
    print('[Failed] Diacritic directory is not found.')
    sys.exit()
print('[  Ok  ] Checked diacritic directory.')

print(f'{" " * 8} Checking input Quran metadata file ...')
if not path.isfile(METADATA_INPUT_FILEPATH):
    print('[Failed] Input Quran metadata file is not found.')
    sys.exit()
print('[  Ok  ] Checked input Quran metadata file.')


# The first three class/functions below were taken from
# https://www.sicara.ai/blog/object-detection-template-matching
class Template:
    """
    A class defining a template
    """
    def __init__(
        self,
        img_path,
        label,
        color,
        matching_threshold=TEMPLATE_MATCHING_THRESHOLD
    ):
        """
        Args:
            img_path (str): path of the template img path
            label (str): the label corresponding to the template
            color (List[int]): the color associated with the label
                (to plot detections)
            matching_threshold (float): the minimum similarity score to consider
                an object is detected by template matching
        """
        self.img_path = img_path
        self.label = label
        self.color = color
        self.template = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        self.template_height, self.template_width = self.template.shape[:2]
        self.matching_threshold = matching_threshold


def compute_iou(
    a,
    b,
    epsilon: int = 1e-5,
    minimum: bool = False
):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero
        minimum: (boolean) Use the minimum area of box `a` or `b` as the divisor

    Returns:
        (float) The Intersect of Union score.
    """
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

    if minimum:
        return area_overlap / min(area_a, area_b)

    area_combined = area_a + area_b - area_overlap
    return area_overlap / (area_combined + epsilon)


def non_max_suppression(
    objects,
    thres: float = 0.5,
    score_key: str = 'match_value'
):
    """
    Filter objects overlapping with IoU over threshold by keeping only the one
        with maximum score.
    Args:
        objects (List[dict]): a list of objects dictionaries, with:
            {score_key} (float): the object score
            {top_left_x} (float): the top-left x-axis coordinate of the object
                bounding box
            {top_left_y} (float): the top-left y-axis coordinate of the object
                bounding box
            {bottom_right_x} (float): the bottom-right x-axis coordinate of the
                object bounding box
            {bottom_right_y} (float): the bottom-right y-axis coordinate of the
                object bounding box
        non_max_suppression_threshold (float): the minimum IoU value used to
            filter overlapping boxes when
            conducting non max suppression.
        score_key (str): score key in objects dicts
    Returns:
        List[dict]: the filtered list of dictionaries.
    """
    def bbox(obj):
        return [
            obj['top_left_x'],
            obj['top_left_y'],
            obj['bottom_right_x'],
            obj['bottom_right_y'],
        ]

    sorted_objects = sorted(objects, key=lambda obj: obj[score_key],
                            reverse=True)
    filtered_objects = []
    for obj in sorted_objects:
        overlafound = False
        for filtered_object in filtered_objects:
            iou = compute_iou(bbox(obj), bbox(filtered_object))
            if iou > thres:
                overlafound = True
                break
        if not overlafound:
            filtered_objects.append(obj)
    return filtered_objects


def find_match(
    img,
    templates,
    apply_nms: bool = True
):
    detections = []
    for template in templates:
        for scale in [100]:  # semi scale-invariant; in percent
            template_img = template.template.copy()
            if scale != 100:
                template_img = cv2.resize(template_img, (
                    int(template_img.shape[1] * scale / 100),
                    int(template_img.shape[0] * scale / 100)
                ))
            for orientation in range(-1, 2):  # semi rotation-invariant
                template_img_ = template_img.copy()
                if orientation != 0:
                    h, w = template_img_.shape[:2]
                    img_center = tuple(np.array([w, h]) / 2)
                    rotation_matrix = cv2.getRotationMatrix2D(
                        img_center, orientation, 1.0
                    )
                    template_img_ = cv2.warpAffine(
                        template_img_, rotation_matrix, (w, h),
                        flags=cv2.INTER_LINEAR
                    )

                template_matching = cv2.matchTemplate(
                    template_img_, img, cv2.TM_CCOEFF_NORMED
                )

                match_locations = np.where(
                    template_matching >= template.matching_threshold
                )

                for (x, y) in zip(match_locations[1], match_locations[0]):
                    detections.append({
                        'top_left_x': x,
                        'top_left_y': y,
                        'bottom_right_x': x + int(template.template_width),
                        'bottom_right_y': y + int(template.template_height),
                        'match_value': template_matching[y, x],
                        'label': template.label,
                        'color': template.color
                    })

    if apply_nms:
        return non_max_suppression(detections, thres=NMS_THRESHOLD)
    else:
        return detections


def get_y_center(bbox):
    return bbox['top_left_y'] + (
        bbox['bottom_right_y'] - bbox['top_left_y']
    ) / 2


print(f'{" " * 8} Parsing input Quran metadata file ...')
ref_suras = etree.parse(METADATA_INPUT_FILEPATH).find('suras').findall('sura')
if ref_suras:
    print('[  Ok  ] Parsed input Quran metadata file.')
else:
    print('[Failed] Input Quran metadata file cannot be parsed.'
          'Please re-download it and then make sure the file is not corrupted.')
    sys.exit()

ref_bboxes = []
ref_idx_sura = SURA_START
ref_idx_aya = AYAH_START
for page_no, page in enumerate(PAGES):
    if VERBOSE_MODE:
        print(f'{" " * 8} Reading page {page_no + 1} ...')
    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    if VERBOSE_MODE:
        print(f'[  Ok  ] Read page {page_no + 1}.')

    # TODO: apply image sharpening
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # img_bgr = cv2.filter2D(img_bgr, -1, kernel)
    cv2.imwrite(
        path.join(QURAN_OUTPUT_DIR, f'{QURAN_TYPE}-resized_{page_no + 1}.'
                  f'{IMAGE_FORMAT}'),
        cv2.resize(img, (int(PAGE_WIDTH * PAGE_SCALE),
                         int(PAGE_HEIGHT * PAGE_SCALE)))
    )
    if page_no < PAGE_START or page_no + 1 > PAGE_END:
        continue

    # Find page border
    if VERBOSE_MODE:
        print(f'{" " * 8} Defining page border ...')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_mod = cv2.GaussianBlur(img_gray, (15, 15), 0)
    _, img_thres = cv2.threshold(
        img_mod,
        0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    img_mod = img_thres.copy()
    if VERBOSE_MODE:
        cv2.imwrite(
            path.join(QURAN_OUTPUT_DIR, f'{QURAN_TYPE}-1_'
                      f'{page_no + 1}.{IMAGE_FORMAT}'), img_mod
        )
    if QURAN_TYPE in ['hafs-jawamee39', 'sample2']:
        if page_no >= PAGE_START + 2 or not SPECIAL_12PAGES:
            img_mod = cv2.Canny(img_mod, 100, 200)
            cv2.floodFill(img_mod, None, (0, 0), 255)  # assume that the
                                                       # background is white
            cv2.floodFill(img_mod, None, (0, 0), 0)
        else:
            cv2.floodFill(img_mod, None, (0, 0), 255)
            img_mod = cv2.bitwise_not(img_mod)
    if QURAN_TYPE in ['hafs-standard39', 'sample']:
        if page_no >= PAGE_START + 2 or not SPECIAL_12PAGES:
            img_mod = cv2.Canny(img_mod, 100, 200)
            cv2.floodFill(img_mod, None, (0, 0), 255)
            cv2.floodFill(img_mod, None, (0, 0), 0)
        else:
            cv2.floodFill(img_mod, None, (0, 0), 255)
            cv2.floodFill(img_mod, None, (PAGE_WIDTH - 1, PAGE_HEIGHT - 1), 255)
            cv2.floodFill(img_mod, None, (0, 0), 0)
            cv2.floodFill(img_mod, None, (0, 0), 255)
            img_mod = cv2.bitwise_not(img_mod)
    if VERBOSE_MODE:
        cv2.imwrite(
            path.join(QURAN_OUTPUT_DIR, f'{QURAN_TYPE}-2_'
                      f'{page_no + 1}.{IMAGE_FORMAT}'),
            img_mod
        )
    bboxes = cv2.findContours(img_mod, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
    bboxes = bboxes[0] if len(bboxes) == 2 else bboxes[1]
    x, y, w, h = cv2.boundingRect(max(bboxes, key=cv2.contourArea))
    PAGE_BORDER = {
        'top_left_x': x,
        'top_left_y': y,
        'bottom_right_x': x + w,
        'bottom_right_y': y + h
    }

    # Remove page border
    if page_no >= PAGE_START + 2 or not SPECIAL_12PAGES:
        img_mod = 0 * np.ones((PAGE_HEIGHT, PAGE_WIDTH), np.uint8)
    else:
        img_mod = 255 * np.ones((PAGE_HEIGHT, PAGE_WIDTH), np.uint8)
    img_mod[
        PAGE_BORDER['top_left_y']:PAGE_BORDER['bottom_right_y'],
        PAGE_BORDER['top_left_x']:PAGE_BORDER['bottom_right_x']
    ] = img_thres[
        PAGE_BORDER['top_left_y']:PAGE_BORDER['bottom_right_y'],
        PAGE_BORDER['top_left_x']:PAGE_BORDER['bottom_right_x']
    ]
    if page_no >= PAGE_START + 2 or not SPECIAL_12PAGES:
        ...
    else:
        cv2.floodFill(img_mod, None, (0, 0), 0)
    if VERBOSE_MODE:
        cv2.imwrite(
            path.join(QURAN_OUTPUT_DIR, f'{QURAN_TYPE}-3_'
                      f'{page_no + 1}.{IMAGE_FORMAT}'), img_mod
        )
        print('[  Ok  ] Defined page border:')
        print(f'{" " * 8} x: {x},')
        print(f'{" " * 8} y: {y},')
        print(f'{" " * 8} w: {w},')
        print(f'{" " * 8} h: {h}.')

    # Find outermost outline
    # TODO: doesn't seem good yet
    if VERBOSE_MODE:
        print(f'{" " * 8} Defining outermost outline ...')
    bboxes = cv2.findContours(img_mod, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
    bboxes = bboxes[0] if len(bboxes) == 2 else bboxes[1]
    x1 = PAGE_WIDTH
    y1 = PAGE_HEIGHT
    x2 = 0
    y2 = 0
    for bbox in bboxes:
        x, y, w, h = cv2.boundingRect(bbox)
        x1 = min(x, x1)
        y1 = min(y, y1)
        x2 = max(x, x2)
        y2 = max(y, y2)
    PAGE_BORDER = {
        'top_left_x': x1,
        'top_left_y': y1,
        'bottom_right_x': x2,
        'bottom_right_y': y2
    }
    if VERBOSE_MODE:
        img_mod = 255 * np.ones((PAGE_HEIGHT, PAGE_WIDTH, 3), np.uint8)
        img_mod[
            PAGE_BORDER['top_left_y']:PAGE_BORDER['bottom_right_y'],
            PAGE_BORDER['top_left_x']:PAGE_BORDER['bottom_right_x']
        ] = img[
            PAGE_BORDER['top_left_y']:PAGE_BORDER['bottom_right_y'],
            PAGE_BORDER['top_left_x']:PAGE_BORDER['bottom_right_x']
        ]
        cv2.imwrite(
            path.join(QURAN_OUTPUT_DIR, f'{QURAN_TYPE}-4_'
                      f'{page_no + 1}.{IMAGE_FORMAT}'), img_mod
        )
        print('[  Ok  ] Defined outermost outline:')
        print(f'{" " * 8} x: {x1},')
        print(f'{" " * 8} y: {y1},')
        print(f'{" " * 8} w: {x2 - x1},')
        print(f'{" " * 8} h: {y2 - y1}.')

    # Find surah, ayah, and bismillah markers
    if VERBOSE_MODE:
        print(f'{" " * 8} Finding markers ...')
    templates = []
    for filename in listdir(DIACRITICS_DIR):
        if filename.endswith(f'.{IMAGE_FORMAT}'):
            if filename.startswith(f'{QURAN_TYPE}-a'):
                templates.append(Template(
                    img_path=path.join(DIACRITICS_DIR, filename),
                    label='ayah', color=(0, 255, 0), matching_threshold=0.5
                ))
            elif filename.startswith(f'{QURAN_TYPE}-b'):
                if page_no == PAGE_START:
                    continue
                templates.append(Template(
                    img_path=path.join(DIACRITICS_DIR, filename),
                    label='bismillah', color=(255, 0, 0),
                    matching_threshold=0.5
                ))
            elif filename.startswith(f'{QURAN_TYPE}-s'):
                templates.append(Template(
                    img_path=path.join(DIACRITICS_DIR, filename),
                    label='surah', color=(0, 0, 255), matching_threshold=0.5
                ))
    markers = find_match(img_gray, templates)
    if VERBOSE_MODE:
        if markers:
            img_mod_ = img.copy()
            ayas = 0
            bismillas = 0
            suras = 0
            for bbox in markers:
                cv2.rectangle(
                    img_mod_,
                    (bbox['top_left_x'], bbox['top_left_y']),
                    (bbox['bottom_right_x'], bbox['bottom_right_y']),
                    bbox['color'], 1
                )
                if bbox['label'] == 'ayah':
                    ayas += 1
                elif bbox['label'] == 'bismillah':
                    bismillas += 1
                else:
                    suras += 1
            cv2.imwrite(
                path.join(QURAN_OUTPUT_DIR, f'{QURAN_TYPE}-5_'
                          f'{page_no + 1}.{IMAGE_FORMAT}'), img_mod_
            )
            print(f'[  Ok  ] Found {len(markers)} markers:')
            print(f'{" " * 8} {ayas} ayahs,')
            print(f'{" " * 8} {bismillas} bismillahs,')
            print(f'{" " * 8} {suras} surahs.')
        else:
            print('[Failed] Cannot found any marker.')
            sys.exit()

    # Find lines
    bbox_sb = [bbox for bbox in markers if bbox['label'] != 'ayah']
    bbox_sb.sort(key=lambda x: x['top_left_y'])
    if VERBOSE_MODE:
        print(f'{" " * 8} Defining lines ...')
    line_numbers = (LINE_NUMBERS[1] if page_no >= PAGE_START + 2 or
                    not SPECIAL_12PAGES else LINE_NUMBERS[0])
    h = (PAGE_BORDER['bottom_right_y'] - PAGE_BORDER['top_left_y']) // line_numbers
    y = PAGE_BORDER['top_left_y']
    bbox_lines = []
    idx_sb = 0
    for idx_line in range(line_numbers):
        label = 'line'
        if idx_sb < len(bbox_sb):
            if y < get_y_center(bbox_sb[idx_sb]) < y + h:
                label = bbox_sb[idx_sb]['label']
                idx_sb += 1

        bbox_lines.append({
            'top_left_x': PAGE_BORDER['top_left_x'],
            'top_left_y': y,
            'bottom_right_x': PAGE_BORDER['bottom_right_x'],
            'bottom_right_y': y + h,
            'label': label
        })
        y += h
    bbox_lines[-1]['bottom_right_y'] = PAGE_BORDER['bottom_right_y']
    if VERBOSE_MODE:
        print('[  Ok  ] Defined lines.')
        for idx_bbox, bbox_line in enumerate(bbox_lines):
            print(f'{" " * 8} Lines {idx_bbox + 1} ({bbox_line["label"]}):')
            print(f'{" " * 10} x1: {bbox_line["top_left_x"]},')
            print(f'{" " * 10} y1: {bbox_line["top_left_y"]},')
            print(f'{" " * 10} x2: {bbox_line["bottom_right_x"]},')
            print(f'{" " * 10} y2: {bbox_line["bottom_right_y"]}.')

    # Refine line (height) bounding boxes
    if page_no >= PAGE_START + 2 or not SPECIAL_12PAGES:
        if VERBOSE_MODE:
            print(f'{" " * 8} Refining line bounding boxes ...')
        idx_sb = 0
        for idx_line in range(len(bbox_lines)):
            if idx_sb < len(bbox_sb):
                if bbox_lines[idx_line]['top_left_y'] < \
                        get_y_center(bbox_sb[idx_sb]) < \
                        bbox_lines[idx_line]['bottom_right_y']:
                    # Resize current line height
                    bbox_lines[idx_line]['bottom_right_y'] = \
                        bbox_lines[idx_line]['top_left_y'] + \
                        bbox_sb[idx_sb]['bottom_right_y'] - \
                        bbox_sb[idx_sb]['top_left_y']
                    # Resize all lines below, but before next surah or bismillah
                    # marker if any
                    line_idxs = []
                    idx_line_ = idx_line + 1
                    while idx_line_ < line_numbers:
                        line_idxs.append(idx_line_)
                        idx_line_ += 1
                        try:
                            if bbox_lines[idx_line_]['label'] \
                                    in ['surah', 'bismillah']:
                                break
                        except:
                            break
                    if line_idxs:
                        y1 = bbox_lines[idx_line]['bottom_right_y']
                        y2 = bbox_lines[line_idxs[-1]]['bottom_right_y']
                        h = (bbox_lines[line_idxs[-1]]['bottom_right_y'] - y1) // len(line_idxs)
                        bbox_lines[line_idxs[-1]]['top_left_y'] = y1
                        for idx in line_idxs:
                            bbox_lines[idx]['top_left_y'] = \
                                bbox_lines[idx - 1]['bottom_right_y']
                            bbox_lines[idx]['bottom_right_y'] = \
                                bbox_lines[idx]['top_left_y'] + h
                        bbox_lines[line_idxs[-1]]['bottom_right_y'] = y2
                    idx_sb += 1
        if VERBOSE_MODE:
            print('[  Ok  ] Refined line bounding boxes.')
            for idx_bbox, bbox_line in enumerate(bbox_lines):
                print(f'{" " * 8} Lines {idx_bbox + 1} ({bbox_line["label"]}):')
                print(f'{" " * 10} x1: {bbox_line["top_left_x"]},')
                print(f'{" " * 10} y1: {bbox_line["top_left_y"]},')
                print(f'{" " * 10} x2: {bbox_line["bottom_right_x"]},')
                print(f'{" " * 10} y2: {bbox_line["bottom_right_y"]}.')

    # Find ayah(s)
    if VERBOSE_MODE:
        print(f'{" " * 8} Defining ayah bounding boxes ...')
    bbox_ayas = [bbox for bbox in markers if bbox['label'] == 'ayah']
    bbox_ayas.sort(key=lambda x: x['top_left_y'])

    bboxes = []
    idx_aya = 0
    for idx_line, bbox_line in enumerate(bbox_lines):
        if bbox_line['label'] != 'line':
            bboxes.append(bbox_line)
            continue

        if VERBOSE_MODE:
            print(f'{" " * 8} Finding ayah(s) on line {idx_line + 1} ...')
        bbox_ayas_ = []
        if len(bbox_ayas) > 0:
            try:
                y_center = get_y_center(bbox_ayas[idx_aya])
                while bbox_line['top_left_y'] < y_center < \
                        bbox_line['bottom_right_y']:
                    bbox_ayas_.append(bbox_ayas[idx_aya])
                    idx_aya += 1
                    try:
                        y_center = get_y_center(bbox_ayas[idx_aya])
                    except:
                        break
            except:
                ...
        if VERBOSE_MODE:
            print(f'[  Ok  ] Found {len(bbox_ayas_)} ayah(s) on line '
                  f'{idx_line + 1}.')

        if page_no >= PAGE_START + 2:
            tolerance = PAGE_BORDER['top_left_x'] + 20 / PAGE_SCALE
        else:
            tolerance = PAGE_BORDER['top_left_x'] + 10 / PAGE_SCALE
        if len(bbox_ayas_) > 0:
            if VERBOSE_MODE:
                print(f'{" " * 8} Segmenting ayah(s) on line {idx_line + 1} ...')
            bbox_ayas_.sort(key=lambda x: x['top_left_x'], reverse=True)
            margin_right = bbox_line['bottom_right_x']
            for idx_aya_, bbox_aya in enumerate(bbox_ayas_):
                bbox_line_ = bbox_line.copy()
                bbox_line_['top_left_x'] = bbox_aya['top_left_x']
                try:
                    if bbox_aya['top_left_x'] < tolerance or (
                        (idx_aya_ + 1) == len(bbox_ayas_) and
                        bbox_lines[idx_line + 1]['label'] != 'line'
                    ):
                        bbox_line_['top_left_x'] = bbox_line['top_left_x']
                except:
                    bbox_line_['top_left_x'] = bbox_line['top_left_x']
                bbox_line_['bottom_right_x'] = margin_right
                bbox_line_['label'] = bbox_aya['label']
                bboxes.append(bbox_line_)
                margin_right = bbox_line_['top_left_x']

            if bbox_line_['top_left_x'] > bbox_line['top_left_x']:
                bbox_line__ = bbox_line.copy()
                bbox_line__['bottom_right_x'] = bbox_line_['top_left_x']
                bboxes.append(bbox_line__)
            if VERBOSE_MODE:
                print(f'[  Ok  ] Segmented all ayah(s) on line {idx_line + 1}.')
        else:
            bboxes.append(bbox_line)
    if VERBOSE_MODE:
        print(f'[  Ok  ] Defined {len(bboxes)} ayah(s) bounding boxes.')

    if not GENERATE_INDEXING and not GENERATE_PREVIEWS:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_aya_segmented = img_rgb.copy()
    for idx_bbox, bbox in enumerate(bboxes):
        if GENERATE_PREVIEWS:
            cv2.rectangle(
                img_aya_segmented,
                (bbox['top_left_x'], int(bbox['top_left_y'])),
                (bbox['bottom_right_x'], int(bbox['bottom_right_y'])),
                (255, 0, 0), 1,
            )
            cv2.putText(
                img_aya_segmented,
                f"{idx_bbox + 1} {bbox['label']}",
                (bbox['top_left_x'] + 2, int(bbox['top_left_y']) + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA,
            )

        if GENERATE_INDEXING:
            if bbox['label'] == 'surah' or \
                    (bbox['label'] == 'bismillah' and  # assume there's no
                     page_no == PAGE_START + 1):       # surah marker on
                                                       # page 2
                aya_no = -1
                ref_idx_sura += 1
                ref_idx_aya = 1
                continue

            aya_no = ref_idx_aya
            if bbox['label'] == 'ayah':
                ref_idx_aya += 1

            ref_bboxes.append({
                'page': page_no + 1,
                'sura': ref_suras[ref_idx_sura - 1].get('index'),
                'aya': aya_no,
                'x1': round(bbox['top_left_x'] * PAGE_SCALE),
                'y1': round(bbox['top_left_y'] * PAGE_SCALE),
                'x2': round(bbox['bottom_right_x'] * PAGE_SCALE),
                'y2': round(bbox['bottom_right_y'] * PAGE_SCALE),
            })

    if GENERATE_PREVIEWS:
        img_bgr = cv2.cvtColor(img_aya_segmented, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            path.join(QURAN_OUTPUT_DIR, f'{QURAN_TYPE}-preview_'
                      f'{page_no + 1}.{IMAGE_FORMAT}'), img_bgr
        )

    print(f'[  Ok  ] Page {page_no + 1} successfully extracted')

if VERBOSE_MODE:
    print(f'{" " * 8} Saving extraction data ...')
if GENERATE_INDEXING:
    keys = ref_bboxes[0].keys()
    with open(METADATA_OUTPUT_FILEPATH, 'w+', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(ref_bboxes)
if VERBOSE_MODE:
    print('[  Ok  ] Saved extraction data.')

print('Alhamdulillaah.')
