import concurrent.futures
import os
import metapub
import requests
import concurrent
from tqdm import tqdm

# Searching for PMID of articles using search query
fetch = metapub.PubMedFetcher()
pmids = fetch.pmids_for_query(query='(perinatal depression[Title]) OR (prenatal depression[Title]) AND (female[Filter])', 
                                 since=2019, 
                                 until=2024, 
                                 retmax=200, 
                                 pmc_only=True)

# Writing list of PMIDs to be downloaded 
with open(r'.\RAG dataset\pmids_list.txt', 'w') as f:
    for pmid in pmids:
        f.write(pmid+'\n')

# Function to download PDFs
def download_pdfs(idx, pmid, dest_path):
    url = metapub.FindIt(pmid).url
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(dest_path, f"{idx}.pdf"), "wb") as f:
            f.write(response.content)


dataset_path = r'.\RAG dataset'
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(download_pdfs, idx, pmid, dataset_path) for idx, pmid in enumerate(pmids)]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        continue
