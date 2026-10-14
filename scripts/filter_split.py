"""
python3 filter_split.py \
    --annotations \
        nt_dataset.json \
        crl_dataset.json \
        nasal_dataset.json \
    --lists \
        list_Hospital-1.txt \
        list_Hospital-2.txt \
    --output nt_crl_nb_splitH1H2.json

python3 filter_split.py \
    --annotations \
        nt_dataset.json \
        crl_dataset.json \
        nasal_dataset.json \
    --lists \
        list_Hospital-1.txt \
        list_Hospital-2.txt \
        list_Hospital-3.txt \
    --output nt_crl_nb_splitH1H2H3.json

python3 filter_split.py \
    --annotations \
        nt_dataset.json \
        crl_dataset.json \
        nasal_dataset.json \
    --lists \
        list_Standard.txt \
    --output nt_crl_nb_Standard.json

python3 filter_split.py \
    --annotations \
        nt_dataset.json \
        crl_dataset.json \
        nasal_dataset.json \
    --lists \
        list_Non-standard.txt \
    --output nt_crl_nb_Non-standard.json
"""
import argparse
import json
from pathlib import Path
from copy import deepcopy


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_filename_lists(list_files):
    filenames = set()

    for list_file in list_files:
        with open(list_file, "r", encoding="utf-8") as f:
            filenames.update(
                line.strip()
                for line in f
                if line.strip()
            )

    return filenames


def filter_and_merge_coco(annotation_files, list_files, output_file):
    # Read and merge all requested filenames
    requested = load_filename_lists(list_files)

    output = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Track IDs and duplicate images
    next_image_id = 1
    next_annotation_id = 1
    next_category_id = 1

    image_id_map = {}
    image_by_filename = {}
    category_by_name = {}

    stats = {
        "input_images": 0,
        "matched_images": 0,
        "input_annotations": 0,
        "matched_annotations": 0,
    }

    for annotation_file in annotation_files:
        coco = load_json(annotation_file)

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])

        stats["input_images"] += len(images)
        stats["input_annotations"] += len(annotations)

        # Map source category IDs to merged category IDs.
        # Category names are used to identify equivalent classes.
        category_id_map = {}

        for category in categories:
            name = category["name"]

            if name not in category_by_name:
                new_category = deepcopy(category)
                new_category["id"] = next_category_id
                category_by_name[name] = next_category_id
                output["categories"].append(new_category)
                next_category_id += 1

            category_id_map[category["id"]] = category_by_name[name]

        # Keep only requested images
        for image in images:
            filename = image["file_name"]

            if filename not in requested:
                continue

            # Reuse existing image ID if filename was already included
            if filename in image_by_filename:
                new_image_id = image_by_filename[filename]
            else:
                new_image = deepcopy(image)
                new_image_id = next_image_id
                new_image["id"] = new_image_id

                output["images"].append(new_image)
                image_by_filename[filename] = new_image_id
                next_image_id += 1
                stats["matched_images"] += 1

            # Source image IDs can collide across input JSONs,
            # so key mappings by input file and original image ID.
            image_id_map[(annotation_file, image["id"])] = new_image_id

        # Keep annotations associated with selected images
        for annotation in annotations:
            old_image_id = annotation["image_id"]
            key = (annotation_file, old_image_id)

            if key not in image_id_map:
                continue

            new_annotation = deepcopy(annotation)
            new_annotation["id"] = next_annotation_id
            new_annotation["image_id"] = image_id_map[key]

            # Remap category ID when category metadata is available
            old_category_id = annotation.get("category_id")
            if old_category_id in category_id_map:
                new_annotation["category_id"] = (
                    category_id_map[old_category_id]
                )

            # Segmentation is optional; preserve it if present.
            # Missing segmentation is valid for bbox-only datasets.
            output["annotations"].append(new_annotation)
            next_annotation_id += 1
            stats["matched_annotations"] += 1

        # Preserve metadata from the first file that supplies it
        if not output["info"] and coco.get("info"):
            output["info"] = deepcopy(coco["info"])

        if not output["licenses"] and coco.get("licenses"):
            output["licenses"] = deepcopy(coco["licenses"])

    # Save merged COCO JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    matched_filenames = set(image_by_filename)
    missing = requested - matched_filenames

    print("\n=== Filtering Summary ===")
    print(f"Input COCO files:       {len(annotation_files)}")
    print(f"Input list files:       {len(list_files)}")
    print(f"Unique requested names: {len(requested)}")
    print(f"Input images:           {stats['input_images']}")
    print(f"Unique matched images:  {stats['matched_images']}")
    print(f"Input annotations:      {stats['input_annotations']}")
    print(f"Annotations kept:       {stats['matched_annotations']}")
    print(f"Categories:             {len(output['categories'])}")
    print(f"Output:                 {output_file}")

    if missing:
        print(f"\nUnmatched filenames: {len(missing)}")
        # for filename in sorted(missing):
            # print(f"  {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Filter and merge multiple COCO JSON files "
            "using multiple filename lists."
        )
    )

    parser.add_argument(
        "--annotations",
        nargs="+",
        required=True,
        help="One or more input COCO JSON files",
    )
    parser.add_argument(
        "--lists",
        nargs="+",
        required=True,
        help="One or more text files containing filenames",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output merged COCO JSON file",
    )

    args = parser.parse_args()

    filter_and_merge_coco(
        annotation_files=args.annotations,
        list_files=args.lists,
        output_file=args.output,
    )
