import os
import json
from .constants import *

def subj_to_tag(subj):
    return subj.strip().lower().replace(' ', '-').replace('(', '[').replace(')', ']').replace(',', '')

def image_path(image):
    return os.path.join(BASE_DIR, image['folder'], image['filename'] + '.jpg')