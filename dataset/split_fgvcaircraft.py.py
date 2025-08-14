import os
import json

ROOT = './FGVCAircraft'
DATASET_DIR = os.path.join(ROOT, 'fgvc-aircraft-2013b', 'data')


def read_data(cname2lab, split_file):
    filepath = os.path.join(DATASET_DIR, split_file)
    items = [x.strip().split(maxsplit=1) for x in open(filepath)]
    items = [(x + '.jpg', cname2lab[y], y) for x, y in items]
    return items


if __name__ == "__main__":
    classnames = []
    with open(os.path.join(DATASET_DIR, 'variants.txt')) as f:
        classnames = [x.strip() for x in f.readlines()]
    cname2lab = {c: i for i, c in enumerate(classnames)}

    train = read_data(cname2lab, 'images_variant_train.txt')
    val = read_data(cname2lab, 'images_variant_val.txt')
    test = read_data(cname2lab, 'images_variant_test.txt')

    with open('split_zhou_FGVCAircraft.json', 'w') as file:
        json.dump({'train': train, 'val': val, 'test': test}, file, indent=4)