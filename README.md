# significancetestXNLP

A collection of tools for NLP research, including a paper crawler for ACL Anthology.

## Features

### ACL Anthology Paper Crawler

Download papers from [ACL Anthology](https://aclanthology.org/) using bibtex entries, URLs, or anthology IDs.

## Installation

1. Clone the repository:

```bash
git clone <Repository-URL>
cd significancetestXNLP
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### ACL Paper Crawler

The `acl_paper_crawler.py` script provides multiple ways to download papers from ACL Anthology:

#### 1. Download from BibTeX file

```bash
python acl_paper_crawler.py --input papers.bib --output downloaded_papers/
```

#### 2. Download from URLs

```bash
python acl_paper_crawler.py --urls "https://aclanthology.org/2020.acl-main.1/,https://aclanthology.org/N19-1423/" --output papers/
```

#### 3. Download from Anthology IDs

```bash
python acl_paper_crawler.py --ids "2020.acl-main.1,N19-1423,P17-1001" --output papers/
```

#### 4. Combine multiple sources

```bash
python acl_paper_crawler.py --input papers.bib --urls "https://aclanthology.org/2021.emnlp-main.1/" --output papers/
```

### Command-line Options

- `--input, -i`: Path to bibtex file containing papers to download
- `--urls, -u`: Comma-separated list of ACL Anthology URLs
- `--ids`: Comma-separated list of ACL Anthology IDs
- `--output, -o`: Output directory for downloaded papers (default: `papers/`)
- `--delay, -d`: Delay between downloads in seconds (default: 1.0)

### Example with Sample File

A sample bibtex file (`example_papers.bib`) is provided. Try it:

```bash
python acl_paper_crawler.py --input example_papers.bib --output papers/
```

## BibTeX Format

The crawler can extract anthology IDs from bibtex entries in several ways:

1. **From URL field**: Extracts anthology ID from `url` field

```bibtex
@inproceedings{example2020,
    title = "Example Paper",
    url = "https://aclanthology.org/2020.acl-main.1/",
    ...
}
```

1. **From DOI field**: Extracts anthology ID from `doi` field

```bibtex
@inproceedings{example2020,
    title = "Example Paper",
    doi = "10.18653/v1/2020.acl-main.1",
    ...
}
```

1. **From citation key**: Uses the citation key if it matches anthology ID pattern

```bibtex
@inproceedings{2020.acl-main.1,
    title = "Example Paper",
    ...
}
```

## Summary of Features

- **Multiple input formats**: Support for bibtex files, URLs, and anthology IDs
- **Respectful crawling**: Built-in delays between downloads to avoid overwhelming the server
- **Smart filename generation**: Uses paper titles in filenames when available
- **Skip existing files**: Automatically skips already downloaded papers
- **Robust error handling**: Continues downloading even if some papers fail
- **Progress tracking**: Shows download progress with statistics

## Requirements

- Python 3.6+
- requests library

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
