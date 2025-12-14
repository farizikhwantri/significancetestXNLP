#!/usr/bin/env python3
"""
ACL Anthology Paper Crawler

This script downloads papers from ACL Anthology (https://aclanthology.org/)
given a list of bibtex entries, URLs, or DOIs.

Usage:
    python acl_paper_crawler.py --input bibtex_file.bib --output papers/
    python acl_paper_crawler.py --urls "https://aclanthology.org/2020.acl-main.1/" --output papers/
    python acl_paper_crawler.py --ids "2020.acl-main.1,2019.naacl-main.2" --output papers/
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    print("Error: requests library not found. Please install it with: pip install requests")
    sys.exit(1)


class ACLPaperCrawler:
    """Crawler for downloading papers from ACL Anthology."""
    
    BASE_URL = "https://aclanthology.org"
    
    # Regex patterns for anthology ID validation
    # Old format: N19-1423, P17-1001, W18-1234 (letter-year-number)
    # New format: 2020.acl-main.1, 2021.findings-emnlp.123 (year.venue-session.paper)
    ANTHOLOGY_ID_PATTERN = r'^([A-Z]\d{2}-\d+|\d{4}\.\w+[\-\.]\w+[\-\.]\d+)$'
    
    def __init__(self, output_dir: str = "papers", delay: float = 1.0):
        """
        Initialize the ACL Paper Crawler.
        
        Args:
            output_dir: Directory to save downloaded papers
            delay: Delay between downloads in seconds (to be respectful)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ACL-Paper-Crawler/1.0 (Educational Purpose)'
        })
    
    def extract_anthology_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract anthology ID from ACL Anthology URL.
        
        Args:
            url: ACL Anthology URL
            
        Returns:
            Anthology ID or None if not found
        """
        # Pattern: https://aclanthology.org/2020.acl-main.1/
        # or https://aclanthology.org/2020.acl-main.1.pdf
        pattern = r'aclanthology\.org/([A-Za-z0-9\-\.]+)'
        match = re.search(pattern, url)
        if match:
            anthology_id = match.group(1)
            # Remove file extensions if present
            anthology_id = anthology_id.replace('.pdf', '').replace('.bib', '')
            return anthology_id
        return None
    
    def extract_anthology_id_from_doi(self, doi: str) -> Optional[str]:
        """
        Extract anthology ID from DOI.
        
        Args:
            doi: DOI string
            
        Returns:
            Anthology ID or None if not found
        """
        # DOI pattern for ACL: 10.18653/v1/2020.acl-main.1
        # or just the anthology ID part
        if doi.startswith('10.18653/v1/'):
            return doi.replace('10.18653/v1/', '')
        elif re.match(r'^[A-Za-z0-9\-\.]+$', doi):
            # Assume it's already an anthology ID
            return doi
        return None
    
    def parse_bibtex_file(self, bibtex_path: str) -> List[Dict[str, str]]:
        """
        Parse bibtex file to extract anthology IDs and metadata.
        
        Args:
            bibtex_path: Path to bibtex file
            
        Returns:
            List of dictionaries containing paper metadata
        """
        papers = []
        
        try:
            with open(bibtex_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading bibtex file: {e}")
            return papers
        
        # Split into individual entries
        entries = re.split(r'@\w+\{', content)
        
        for entry in entries[1:]:  # Skip first empty split
            paper_info = {}
            
            # Extract citation key (anthology ID is often here)
            citation_key_match = re.match(r'([^,]+),', entry)
            if citation_key_match:
                citation_key = citation_key_match.group(1).strip()
                paper_info['citation_key'] = citation_key
            
            # Extract URL field
            url_match = re.search(r'url\s*=\s*["{]([^"}]+)["}]', entry, re.IGNORECASE)
            if url_match:
                url = url_match.group(1).strip()
                paper_info['url'] = url
                anthology_id = self.extract_anthology_id_from_url(url)
                if anthology_id:
                    paper_info['anthology_id'] = anthology_id
            
            # Extract DOI field
            doi_match = re.search(r'doi\s*=\s*["{]([^"}]+)["}]', entry, re.IGNORECASE)
            if doi_match:
                doi = doi_match.group(1).strip()
                paper_info['doi'] = doi
                if 'anthology_id' not in paper_info:
                    anthology_id = self.extract_anthology_id_from_doi(doi)
                    if anthology_id:
                        paper_info['anthology_id'] = anthology_id
            
            # Extract title (handle nested braces and quotes)
            title_match = re.search(r'title\s*=\s*["{](.+?)["}]\s*[,\n}]', entry, re.IGNORECASE | re.DOTALL)
            if title_match:
                title = title_match.group(1).strip()
                # Remove outer LaTeX-style braces if present (non-greedy to avoid nested issues)
                title = re.sub(r'^\{(.+?)\}$', r'\1', title)
                paper_info['title'] = title
            
            # If we have an anthology ID, add to list
            if 'anthology_id' in paper_info:
                papers.append(paper_info)
            # If citation key looks like an anthology ID, use it
            elif 'citation_key' in paper_info and re.match(self.ANTHOLOGY_ID_PATTERN, paper_info['citation_key']):
                paper_info['anthology_id'] = paper_info['citation_key']
                papers.append(paper_info)
        
        return papers
    
    def download_paper(self, anthology_id: str, title: Optional[str] = None) -> bool:
        """
        Download a paper from ACL Anthology.
        
        Args:
            anthology_id: ACL Anthology ID (e.g., "2020.acl-main.1")
            title: Optional paper title for filename
            
        Returns:
            True if successful, False otherwise
        """
        pdf_url = f"{self.BASE_URL}/{anthology_id}.pdf"
        
        # Create filename
        if title:
            # Clean title for filename - remove only filesystem-unsafe characters
            clean_title = re.sub(r'[/\\:*?"<>|]', '_', title)
            # Remove extra whitespace and limit length
            clean_title = re.sub(r'\s+', '_', clean_title)[:100]
            filename = f"{anthology_id}_{clean_title}.pdf"
        else:
            filename = f"{anthology_id}.pdf"
        
        output_path = self.output_dir / filename
        
        # Skip if already exists
        if output_path.exists():
            print(f"  Already exists: {filename}")
            return True
        
        try:
            print(f"  Downloading: {anthology_id}")
            response = self.session.get(pdf_url, timeout=30)
            
            if response.status_code == 200:
                # Verify it's a PDF by checking content type and magic number
                content_type = response.headers.get('Content-Type', '')
                is_pdf_content_type = content_type.startswith('application/pdf')
                # Check for PDF magic number (%PDF)
                is_pdf_magic = response.content[:4] == b'%PDF'
                
                if is_pdf_content_type or is_pdf_magic:
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    print(f"  ✓ Saved: {filename}")
                    return True
                else:
                    print(f"  ✗ Not a PDF: {anthology_id}")
                    return False
            elif response.status_code == 404:
                print(f"  ✗ Not found: {anthology_id}")
                return False
            else:
                print(f"  ✗ Error {response.status_code}: {anthology_id}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Network error for {anthology_id}: {e}")
            return False
        except Exception as e:
            print(f"  ✗ Unexpected error for {anthology_id}: {e}")
            return False
    
    def crawl_from_bibtex(self, bibtex_path: str) -> Dict[str, int]:
        """
        Crawl and download papers from a bibtex file.
        
        Args:
            bibtex_path: Path to bibtex file
            
        Returns:
            Dictionary with statistics (total, success, failed)
        """
        print(f"Parsing bibtex file: {bibtex_path}")
        papers = self.parse_bibtex_file(bibtex_path)
        
        if not papers:
            print("No papers found in bibtex file with valid anthology IDs")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        print(f"Found {len(papers)} papers with anthology IDs")
        print(f"Downloading to: {self.output_dir}")
        print()
        
        stats = {'total': len(papers), 'success': 0, 'failed': 0}
        
        for i, paper in enumerate(papers, 1):
            anthology_id = paper.get('anthology_id')
            title = paper.get('title')
            
            print(f"[{i}/{len(papers)}] {anthology_id}")
            if title:
                print(f"  Title: {title[:80]}...")
            
            success = self.download_paper(anthology_id, title)
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
            
            # Be respectful with delays
            if i < len(papers):
                time.sleep(self.delay)
        
        return stats
    
    def crawl_from_urls(self, urls: List[str]) -> Dict[str, int]:
        """
        Crawl and download papers from a list of URLs.
        
        Args:
            urls: List of ACL Anthology URLs
            
        Returns:
            Dictionary with statistics (total, success, failed)
        """
        anthology_ids = []
        for url in urls:
            anthology_id = self.extract_anthology_id_from_url(url)
            if anthology_id:
                anthology_ids.append(anthology_id)
            else:
                print(f"Warning: Could not extract anthology ID from: {url}")
        
        if not anthology_ids:
            print("No valid anthology IDs found in URLs")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        return self.crawl_from_ids(anthology_ids)
    
    def crawl_from_ids(self, anthology_ids: List[str]) -> Dict[str, int]:
        """
        Crawl and download papers from a list of anthology IDs.
        
        Args:
            anthology_ids: List of ACL Anthology IDs
            
        Returns:
            Dictionary with statistics (total, success, failed)
        """
        print(f"Found {len(anthology_ids)} anthology IDs")
        print(f"Downloading to: {self.output_dir}")
        print()
        
        stats = {'total': len(anthology_ids), 'success': 0, 'failed': 0}
        
        for i, anthology_id in enumerate(anthology_ids, 1):
            print(f"[{i}/{len(anthology_ids)}] {anthology_id}")
            
            success = self.download_paper(anthology_id)
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
            
            # Be respectful with delays
            if i < len(anthology_ids):
                time.sleep(self.delay)
        
        return stats


def main():
    """Main function to handle command-line interface."""
    parser = argparse.ArgumentParser(
        description='Download papers from ACL Anthology',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from bibtex file
  %(prog)s --input papers.bib --output downloaded_papers/
  
  # Download from URLs
  %(prog)s --urls "https://aclanthology.org/2020.acl-main.1/" --output papers/
  
  # Download from anthology IDs
  %(prog)s --ids "2020.acl-main.1,2019.naacl-main.2" --output papers/
  
  # Download from multiple sources with custom delay
  %(prog)s --input papers.bib --urls "https://aclanthology.org/2021.emnlp-main.1/" --delay 2.0
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to bibtex file containing papers to download'
    )
    
    parser.add_argument(
        '--urls', '-u',
        type=str,
        help='Comma-separated list of ACL Anthology URLs'
    )
    
    parser.add_argument(
        '--ids',
        type=str,
        help='Comma-separated list of ACL Anthology IDs (e.g., "2020.acl-main.1,2019.naacl-main.2")'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='papers',
        help='Output directory for downloaded papers (default: papers/)'
    )
    
    parser.add_argument(
        '--delay', '-d',
        type=float,
        default=1.0,
        help='Delay between downloads in seconds (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Check if at least one input method is provided
    if not any([args.input, args.urls, args.ids]):
        parser.error("At least one of --input, --urls, or --ids must be provided")
    
    # Create crawler
    crawler = ACLPaperCrawler(output_dir=args.output, delay=args.delay)
    
    total_stats = {'total': 0, 'success': 0, 'failed': 0}
    
    # Process bibtex file
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Bibtex file not found: {args.input}")
            sys.exit(1)
        
        print("=" * 70)
        print("Processing bibtex file")
        print("=" * 70)
        stats = crawler.crawl_from_bibtex(args.input)
        for key in total_stats:
            total_stats[key] += stats[key]
        print()
    
    # Process URLs
    if args.urls:
        urls = [url.strip() for url in args.urls.split(',')]
        print("=" * 70)
        print("Processing URLs")
        print("=" * 70)
        stats = crawler.crawl_from_urls(urls)
        for key in total_stats:
            total_stats[key] += stats[key]
        print()
    
    # Process anthology IDs
    if args.ids:
        ids = [id.strip() for id in args.ids.split(',')]
        print("=" * 70)
        print("Processing anthology IDs")
        print("=" * 70)
        stats = crawler.crawl_from_ids(ids)
        for key in total_stats:
            total_stats[key] += stats[key]
        print()
    
    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total papers:      {total_stats['total']}")
    print(f"Successfully downloaded: {total_stats['success']}")
    print(f"Failed:            {total_stats['failed']}")
    print(f"Output directory:  {crawler.output_dir.absolute()}")
    print()
    
    if total_stats['failed'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
