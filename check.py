import json
import os
import shutil
import argparse

def find_differences(struct1, struct2, path=""):
    """
    Recursively find differences between two JSON structures.
    Returns a list of differences.
    """
    diffs = []

    if isinstance(struct1, dict) and isinstance(struct2, dict):
        for key in struct1.keys():
            new_path = f"{path}.{key}" if path else key
            if key not in struct2:
                diffs.append(f"Missing key in second structure: {new_path}")
            else:
                diffs.extend(find_differences(struct1[key], struct2[key], new_path))
        for key in struct2.keys():
            new_path = f"{path}.{key}" if path else key
            if key not in struct1:
                diffs.append(f"Extra key in second structure: {new_path}")
    elif isinstance(struct1, int) and isinstance(struct2, int):
        if struct1 != struct2:
            diffs.append(f"Different list lengths at {path}: {struct1} vs. {struct2}")
    else:
        # If we have other types, we can extend the comparisons here.
        pass

    return diffs

def json_structure(json_obj):
    """
    Returns a simplified representation of the JSON's structure.
    """
    if isinstance(json_obj, dict):
        return {key: json_structure(value) for key, value in json_obj.items()}
    elif isinstance(json_obj, list):
        # Check if list contains dictionaries
        if all(isinstance(item, dict) for item in json_obj):
            return {f"item_{idx}": json_structure(item) for idx, item in enumerate(json_obj)}
        else:
            return len(json_obj)
    else:
        return None

def gather_mismatched_images(mismatched_files, base_dir):
    """
    Gathers the mismatched images into a new directory.
    """
    dest_dir = os.path.join(base_dir, "..", "mismatched")
    os.makedirs(dest_dir, exist_ok=True)

    for file_number in mismatched_files:
        orig_img = os.path.join(base_dir, "..", "train_img", f"{file_number}.jpg")
        rendered_img = os.path.join(base_dir, "..", "train_openpose_img", f"{file_number}_rendered.jpg")

        shutil.copy2(orig_img, dest_dir)
        shutil.copy2(rendered_img, dest_dir)

def compare_jsons(dir_path):
    """
    Compares the structure of multiple JSON files in a directory.
    """
    structures = []
    filenames = [f for f in os.listdir(dir_path) if f.endswith('.json')]

    # Store the structure of each JSON file
    for filename in filenames:
        with open(os.path.join(dir_path, filename), 'r') as f:
            content = json.load(f)
            structures.append(json_structure(content))

    first_structure = structures[0]
    differing_files = []

    # Compare the structures
    for filename, structure in zip(filenames[1:], structures[1:]):
        diffs = find_differences(first_structure, structure)
        if diffs:
            differing_files.append((filename, diffs))
    
    if differing_files:
        print(f"The following files have different structures:")
        for filename, diffs in differing_files:
            print(f"\nDifferences in {filename}:")
            for diff in diffs:
                print(f"  - {diff}")
        mismatched_files = [fname.split('_')[0] for fname, _ in differing_files]
        return mismatched_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare JSON structures and optionally gather mismatched images.")
    parser.add_argument("directory", type=str, help="Directory to read json files from")
    parser.add_argument('--gather', action='store_true', help="Gather mismatched images into the 'mismatched' directory.")
    args = parser.parse_args()

    
    mismatched_files = compare_jsons(args.directory)
    if args.gather and mismatched_files:
      print("Moving mismatched files to new directory")
      gather_mismatched_images(mismatched_files, args.directory)
