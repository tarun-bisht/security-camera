def read_label_pbtxt(label_path: str) -> dict:
    with open(label_path, "r") as label_file:
        lines = label_file.readlines()
        labels = {}
        for row, content in enumerate(lines):
            labels[row] = {"id": row, "name": content.strip()}
    return labels
