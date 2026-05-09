PAPERS_SHEET = "List of Papers"
DATASETS_SHEET = "List of Datasets"
BIBTEX_SHEET = "Bibtex"
REVIEW_SHEET = "Review Needed"
EVIDENCE_SHEET = "Evidence"
PAPER_DATASET_LINKS_SHEET = "Paper Dataset Links"

PAPERS_COLUMNS = [
    "Paper Title",
    "Authors",
    "Conference",
    "Year",
    "Datasets",
    "Bibtex Citation Key",
]

DATASETS_COLUMNS = [
    "dataset name",
    "bibtex citation key",
    "OSI layer at which dataset is measured",
    "modality(ies)",
    "Availaibility (open?)",
    "Availability Annotations",
    "Collection environment",
    "Number of Papers using Dataset",
]

BIBTEX_COLUMNS = [
    "bibtex citation key",
    "DOI version of key",
    "bibtex citation",
]

REVIEW_COLUMNS = [
    "Item Type",
    "Paper Title",
    "Dataset Name",
    "Field",
    "Suggested Value",
    "Confidence",
    "Review Reason",
    "Evidence",
    "Source URL",
]

EVIDENCE_COLUMNS = [
    "Claim ID",
    "Paper Title",
    "Dataset Name",
    "Claim Type",
    "Claim Value",
    "Evidence Text",
    "Evidence Source",
    "Source URL",
    "Checked At",
    "Confidence",
]

PAPER_DATASET_LINKS_COLUMNS = [
    "Paper Title",
    "Bibtex Citation Key",
    "Dataset Name",
    "Relationship Type",
    "Confidence",
    "Evidence",
    "Review Needed",
]

SHEETS = {
    PAPERS_SHEET: PAPERS_COLUMNS,
    DATASETS_SHEET: DATASETS_COLUMNS,
    BIBTEX_SHEET: BIBTEX_COLUMNS,
    REVIEW_SHEET: REVIEW_COLUMNS,
    EVIDENCE_SHEET: EVIDENCE_COLUMNS,
    PAPER_DATASET_LINKS_SHEET: PAPER_DATASET_LINKS_COLUMNS,
}
