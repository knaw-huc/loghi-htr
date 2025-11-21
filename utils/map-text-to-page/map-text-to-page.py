import argparse
import xml.etree.ElementTree as ET
import editdistance
import numpy as np


def parse_pagexml(file_path):
    """
    Parse the PageXML file and extract text lines.
    Normalizes missing/empty Unicode to empty string.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    namespace = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    text_lines = []

    for text_line in root.findall(".//ns:TextLine", namespace):
        line_id = text_line.get("id")
        text_equiv = text_line.find("./ns:TextEquiv/ns:Unicode", namespace)
        # Normalize to empty string if missing or None, and strip whitespace
        text_value = ""
        if text_equiv is not None and text_equiv.text is not None:
            text_value = text_equiv.text.strip()
        text_lines.append((line_id, text_value))

    return tree, root, namespace, text_lines


def read_ground_truth(file_path):
    """
    Read the ground truth text file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def best_substring_distance(gt: str, text: str, pad: int = 10) -> int:
    """
    Compute minimal edit distance between gt and any substring of text.
    The window length is searched around len(gt) with +/- pad.
    """
    gt = gt or ""
    text = text or ""
    if not gt or not text:
        return editdistance.eval(gt, text)

    n = len(text)
    L = len(gt)

    # Baseline: whole text
    best = editdistance.eval(gt, text)

    # Window lengths around L within bounds
    min_w = max(1, L - pad)
    max_w = min(n, L + pad)
    for w in range(min_w, max_w + 1):
        # Iterate all substrings of length w
        for start in range(0, n - w + 1):
            s = text[start:start + w]
            d = editdistance.eval(gt, s)
            if d < best:
                best = d
                if best == 0:
                    return 0  # Early exit

    return best


def best_window_concat_substring_distance(
        gt: str,
        page_texts: list[str],
        start_index: int,
        max_span: int = 2,
        pad: int = 10,
        sep: str = " "
) -> int:
    """
    Compute minimal edit distance between gt and any substring within the
    concatenation of up to `max_span` adjacent PageXML lines starting at `start_index`.
    """
    gt = gt or ""
    if not gt:
        # Empty GT matches empty substring with 0
        return 0

    n = len(page_texts)
    best = float("inf")

    # Try spans: 1..max_span, staying within bounds
    for span in range(1, max_span + 1):
        end = start_index + span
        if end > n:
            break
        concat_text = " ".join((page_texts[k] or "") for k in range(start_index, end)).strip()
        if not concat_text:
            # Still compute to keep consistent behavior
            d = editdistance.eval(gt, concat_text)
        else:
            d = best_substring_distance(gt, concat_text, pad=pad)
        if d < best:
            best = d
            if best == 0:
                return 0  # Early exit

    return int(best)


def map_textlines(
        pagexml_lines,
        ground_truth_lines,
        allow_concatenated_to_partial: bool = False,
        allow_partial_to_concatenated: bool = False,
        max_concat_span: int = 2,
        pad: int = 10
):
    """
    Map ground truth lines to PageXML lines to minimize total Levenshtein distance.

    - When `allow_concatenated_to_partial` is True, match GT lines to the best substring
      of a single PageXML line (GT may be concatenated; page line is partial).
    - When `allow_partial_to_concatenated` is True, match GT lines to the best substring
      within the concatenation of up to `max_concat_span` adjacent PageXML lines.
    - If both are True, the minimum of applicable strategies is used.
    """
    num_gt = len(ground_truth_lines)
    num_pagexml = len(pagexml_lines)

    if num_gt == 0 or num_pagexml == 0:
        # Nothing to map
        mapping = [(gt, None, None, None) for gt in ground_truth_lines]
        return mapping

    # Prepare plain texts for windowed concatenation computation
    page_texts = [(pl if pl is not None else "") for _, pl in pagexml_lines]

    # Create a cost matrix for Levenshtein distances
    cost_matrix = np.full((num_gt, num_pagexml), float("inf"), dtype=np.float32)
    for i, gt_line in enumerate(ground_truth_lines):
        gt = gt_line if gt_line is not None else ""
        for j, (line_id, page_line) in enumerate(pagexml_lines):
            pl = page_line if page_line is not None else ""

            # Base exact distance
            candidates = [editdistance.eval(gt, pl)]

            # Allow GT concatenated -> partial page substring
            if allow_concatenated_to_partial:
                candidates.append(best_substring_distance(gt, pl, pad=pad))

            # Allow GT partial -> concatenated adjacent page lines
            if allow_partial_to_concatenated:
                candidates.append(
                    best_window_concat_substring_distance(
                        gt,
                        page_texts,
                        start_index=j,
                        max_span=max_concat_span,
                        pad=pad,
                        sep=" "
                    )
                )

            cost_matrix[i, j] = float(min(candidates))

    # Solve the assignment problem using Hungarian algorithm
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create the mapping
    mapping = []
    for i, j in zip(row_ind, col_ind):
        gt_line = ground_truth_lines[i]
        line_id, page_line = pagexml_lines[j]
        distance = float(cost_matrix[i, j])
        mapping.append((gt_line, line_id, page_line, distance))

    # Add unmatched ground truth lines
    matched_gt_indices = set(row_ind)
    for i, gt_line in enumerate(ground_truth_lines):
        if i not in matched_gt_indices:
            mapping.append((gt_line, None, None, None))

    return mapping


def update_pagexml_with_matches(tree, root, namespace, mapping):
    """
    Update the PageXML file with the best matches from the ground truth.
    """
    for gt_line, line_id, page_line, _ in mapping:
        if line_id is not None:
            text_line = root.find(f".//ns:TextLine[@id='{line_id}']", namespace)
            if text_line is not None:
                text_equiv = text_line.find(".//ns:TextEquiv/ns:Unicode", namespace)
                if text_equiv is not None:
                    text_equiv.text = gt_line  # Replace with the best match

    return tree


def main():
    parser = argparse.ArgumentParser(description="Map ground truth text to PageXML text lines.")
    parser.add_argument("--pagexml_path", help="Path to the PageXML file.", required=True)
    parser.add_argument("--ground_truth_path", help="Path to the ground truth text file.", required=True)
    parser.add_argument("--output_path", help="Path to the output file.", required=True)
    parser.add_argument(
        "--allow_concatenated_to_partial",
        action="store_true",
        help="When set, allows GT concatenated lines to match partial substrings of a single PageXML line."
    )
    parser.add_argument(
        "--allow_partial_to_concatenated",
        action="store_true",
        help="When set, allows GT partial lines to match substrings within concatenations of adjacent PageXML lines."
    )
    parser.add_argument(
        "--max_concat_span",
        type=int,
        default=2,
        help="Max number of adjacent PageXML lines to concatenate when --allow_partial_to_concatenated is set."
    )
    parser.add_argument(
        "--substring_pad",
        type=int,
        default=10,
        help="Substring length search padding around len(GT) for partial matching."
    )
    args = parser.parse_args()

    # Parse PageXML and ground truth
    tree, root, namespace, pagexml_lines = parse_pagexml(args.pagexml_path)
    ground_truth_lines = read_ground_truth(args.ground_truth_path)

    # Map text lines
    mapping = map_textlines(
        pagexml_lines,
        ground_truth_lines,
        allow_concatenated_to_partial=args.allow_concatenated_to_partial,
        allow_partial_to_concatenated=args.allow_partial_to_concatenated,
        max_concat_span=args.max_concat_span,
        pad=args.substring_pad,
    )
    # Count lines without match
    count_lines_without_match = sum(1 for _, line_id, _, _ in mapping if line_id is None)

    # Update PageXML with matches
    updated_tree = update_pagexml_with_matches(tree, root, namespace, mapping)

    # Write the updated PageXML to the output file
    updated_tree.write(args.output_path, encoding="utf-8", xml_declaration=True)

    # Write results to output
    total_distance = 0.0
    for gt_line, line_id, page_line, distance in mapping:
        if distance is not None and distance >= 0:
            print(f"GT: {gt_line}\n")
            print(f"Matched Line ID: {line_id}\n")
            print(f"Matched Text: {page_line}\n")
            print(f"Levenshtein Distance: {distance}\n")
            print("\n")
            total_distance += distance

    print(f'lines without match: {count_lines_without_match}\n')
    print(f'Total Levenshtein Distance: {int(total_distance)}\n')


if __name__ == "__main__":
    main()