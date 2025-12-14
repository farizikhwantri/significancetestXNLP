#!/usr/bin/env python3
"""
Example usage of ACL Paper Crawler as a Python module.

This demonstrates how to use the ACLPaperCrawler class programmatically
in your own Python scripts.
"""

from acl_paper_crawler import ACLPaperCrawler


def example_basic_usage():
    """Basic usage example."""
    print("Example 1: Basic usage with anthology IDs")
    print("=" * 70)
    
    # Create a crawler instance
    crawler = ACLPaperCrawler(output_dir="my_papers", delay=1.0)
    
    # Download papers by anthology IDs
    anthology_ids = ["2020.acl-main.1", "N19-1423", "P17-1001"]
    stats = crawler.crawl_from_ids(anthology_ids)
    
    print(f"\nResults: {stats}")
    print()


def example_bibtex_usage():
    """Example using a bibtex file."""
    print("Example 2: Using a bibtex file")
    print("=" * 70)
    
    # Create a crawler instance
    crawler = ACLPaperCrawler(output_dir="papers_from_bibtex", delay=1.5)
    
    # Download papers from bibtex file
    stats = crawler.crawl_from_bibtex("example_papers.bib")
    
    print(f"\nResults: {stats}")
    print()


def example_url_usage():
    """Example using URLs."""
    print("Example 3: Using ACL Anthology URLs")
    print("=" * 70)
    
    # Create a crawler instance
    crawler = ACLPaperCrawler(output_dir="papers_from_urls", delay=1.0)
    
    # Download papers from URLs
    urls = [
        "https://aclanthology.org/2020.acl-main.1/",
        "https://aclanthology.org/N19-1423/",
    ]
    stats = crawler.crawl_from_urls(urls)
    
    print(f"\nResults: {stats}")
    print()


def example_parsing_only():
    """Example showing how to parse bibtex without downloading."""
    print("Example 4: Parsing bibtex file without downloading")
    print("=" * 70)
    
    # Create a crawler instance
    crawler = ACLPaperCrawler()
    
    # Parse bibtex file to extract paper information
    papers = crawler.parse_bibtex_file("example_papers.bib")
    
    print(f"Found {len(papers)} papers:\n")
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper.get('anthology_id')}")
        print(f"   Title: {paper.get('title', 'N/A')}")
        print(f"   URL: {paper.get('url', 'N/A')}")
        print()


def example_extraction_utils():
    """Example showing utility functions for extracting anthology IDs."""
    print("Example 5: Extracting anthology IDs from various sources")
    print("=" * 70)
    
    crawler = ACLPaperCrawler()
    
    # Extract from URL
    url = "https://aclanthology.org/2020.acl-main.1/"
    anthology_id = crawler.extract_anthology_id_from_url(url)
    print(f"From URL: {url}")
    print(f"  -> Anthology ID: {anthology_id}\n")
    
    # Extract from DOI
    doi = "10.18653/v1/N19-1423"
    anthology_id = crawler.extract_anthology_id_from_doi(doi)
    print(f"From DOI: {doi}")
    print(f"  -> Anthology ID: {anthology_id}\n")


if __name__ == "__main__":
    # Run all examples
    # Note: The actual downloads will fail in the sandbox environment
    # but will work in a real environment with internet access
    
    example_parsing_only()
    example_extraction_utils()
    
    # Uncomment to run download examples (requires internet access):
    # example_basic_usage()
    # example_bibtex_usage()
    # example_url_usage()
