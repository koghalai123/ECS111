import pandas as pd
import requests, sys, json
import numpy as np
from io import StringIO
import re


def get_uniprot_proteins(taxonomy_id):
    proteins = []
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"(taxonomy_id:{taxonomy_id})",  # Added parentheses for exact match
        "format": "json",
        "fields": "accession,id,protein_name,gene_names,organism_name,organism_id,length",
        "size": 500
    }
    
    while True:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Verify we're getting the right taxonomy
        for protein in data["results"]:
            if str(protein.get("organism", {}).get("taxonId")) != str(taxonomy_id):
                print(f"Warning: Found protein with wrong taxonomy ID: {protein.get('primaryAccession')} "
                      f"(Expected: {taxonomy_id}, Got: {protein.get('organism', {}).get('taxonId')})")
        
        proteins.extend(data["results"])
        
        if "next" not in data.get("links", {}):
            break
            
        url = data["links"]["next"]
        params = {}
        
    return proteins

def flatten_protein_data(protein):
    """Extract relevant fields from the nested protein data"""
    return {
        'Accession': protein.get('primaryAccession'),
        'ID': protein.get('uniProtkbId'),
        'Protein Name': protein.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value'),
        'Gene Names': ', '.join([gene.get('geneName', {}).get('value') for gene in protein.get('genes', []) 
                               if gene.get('geneName', {}).get('value')]),
        'Organism': protein.get('organism', {}).get('scientificName'),
        'Taxonomy ID': protein.get('organism', {}).get('taxonId'),
        'Length': protein.get('sequence', {}).get('length')
    }


def parse_fasta_header(header):
    # Regular expression to extract fields from the FASTA header
    match = re.match(
        r'(?P<db>\w+)\|(?P<accession>\w+)\|(?P<entry_name>\S+)\s+(?P<protein_name>.+?)\s+OS=(?P<organism>.+?)\s+OX=(?P<tax_id>\d+)\s+GN=(?P<gene_name>\S+)\s+PE=(?P<pe>\d+)\s+SV=(?P<sv>\d+)',
        header
    )
    if match:
        return match.groupdict()
    else:
        # If the header doesn't match, return minimal info
        return {"raw_header": header}

def uniprot_stream(taxonomy_id):
    #taxonomy_id = 425104
    url = f"https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28%28taxonomy_id%3A{taxonomy_id}%29%29"
    response = requests.get(url)
    response.raise_for_status()
    fasta_data = response.text

    # Parse FASTA into a DataFrame
    records = []
    entry = {}
    for line in StringIO(fasta_data):
        line = line.strip()
        if line.startswith(">"):
            if entry:
                records.append(entry)
            header = line[1:]
            parsed = parse_fasta_header(header)
            entry = parsed
            entry["Sequence"] = ""
        elif line:
            entry["Sequence"] += line
    if entry:
        records.append(entry)

    df = pd.DataFrame(records)
    return df

def uniprot_search(lowTempEntries,medTempEntries):
    proteinDF = pd.DataFrame()

    for i in range(lowTempEntries.shape[0]):
        taxonomy_id = str(lowTempEntries['taxonomy_id'].iloc[i])
        print(f"Querying proteins for taxonomy ID: {taxonomy_id}")
        proteins = get_uniprot_proteins(taxonomy_id)

        # Convert to DataFrame
        if proteins:
            df = pd.DataFrame([flatten_protein_data(p) for p in proteins])
            print(f"\nFound {len(df)} proteins for taxonomy ID {taxonomy_id}")
            print(df.head())
            
            # Verify all results match our taxonomy ID
            mismatches = df[df['Taxonomy ID'].astype(str) != str(taxonomy_id)]
            proteinDF = pd.concat([proteinDF, df], ignore_index=True)
            if not mismatches.empty:
                print(f"\nWarning: Found {len(mismatches)} proteins with incorrect taxonomy IDs:")
                print(mismatches[['Accession', 'Taxonomy ID']])
        else:
            print("No proteins found for this taxonomy ID")

    proteinDF.to_csv('lowTempProteins.csv', index=False)



    proteinDF = pd.DataFrame()

    for i in range(np.min([medTempEntries.shape[0],50])):
        taxonomy_id = str(medTempEntries['taxonomy_id'].iloc[i])
        print(f"Querying proteins for taxonomy ID: {taxonomy_id}")
        proteins = get_uniprot_proteins(taxonomy_id)

        # Convert to DataFrame
        if proteins:
            df = pd.DataFrame([flatten_protein_data(p) for p in proteins])
            print(f"\nFound {len(df)} proteins for taxonomy ID {taxonomy_id}")
            print(df.head())
            
            # Verify all results match our taxonomy ID
            mismatches = df[df['Taxonomy ID'].astype(str) != str(taxonomy_id)]
            proteinDF = pd.concat([proteinDF, df], ignore_index=True)
            if not mismatches.empty:
                print(f"\nWarning: Found {len(mismatches)} proteins with incorrect taxonomy IDs:")
                print(mismatches[['Accession', 'Taxonomy ID']])
        else:
            print("No proteins found for this taxonomy ID")

    proteinDF.to_csv('medTempProteins.csv', index=False)


file_path = 'FinalProject/200617_TEMPURA.csv' 
data = pd.read_csv(file_path)
unique_elements = data['Topt_ave'].unique() 

lowTempEntries = data[(data['Topt_ave'] <= 10)]

medTempEntries = data[(data['Topt_ave'] >= 20) & (data['Topt_ave'] <= 40)]
#uniprot_search(lowTempEntries,medTempEntries)


proteinDF = pd.DataFrame()

for i in range(lowTempEntries.shape[0]):
    taxonomy_id = str(lowTempEntries['taxonomy_id'].iloc[i])
    print(f"Querying proteins for taxonomy ID: {taxonomy_id}")
    proteins = uniprot_stream(taxonomy_id)

    # Convert to DataFrame
    if not proteins.empty:
        df = proteins  # proteins is already a DataFrame
        print(f"\nFound {len(df)} proteins for taxonomy ID {taxonomy_id}")
        print(df.head())
        
        # Verify all results match our taxonomy ID
        mismatches = df[df['tax_id'].astype(str) != str(taxonomy_id)]
        proteinDF = pd.concat([proteinDF, df], ignore_index=True)
        if not mismatches.empty:
            print(f"\nWarning: Found {len(mismatches)} proteins with incorrect taxonomy IDs:")
            print(mismatches[['accession', 'tax_id']])
    else:
        print("No proteins found for this taxonomy ID")
proteinDF.to_csv('lowTempProteins.csv', index=False)

proteinDF = pd.DataFrame()

for i in range(np.min([medTempEntries.shape[0],50])):
    taxonomy_id = str(medTempEntries['taxonomy_id'].iloc[i])
    print(f"Querying proteins for taxonomy ID: {taxonomy_id}")
    proteins = uniprot_stream(taxonomy_id)

    # Convert to DataFrame
    if not proteins.empty:
        df = proteins  # proteins is already a DataFrame
        print(f"\nFound {len(df)} proteins for taxonomy ID {taxonomy_id}")
        print(df.head())
        
        # Verify all results match our taxonomy ID
        #mismatches = df[df['tax_id'].astype(str) != str(taxonomy_id)]
        proteinDF = pd.concat([proteinDF, df], ignore_index=True)
        #if not mismatches.empty:
            #print(f"\nWarning: Found {len(mismatches)} proteins with incorrect taxonomy IDs:")
            #print(mismatches[['accession', 'tax_id']])
    else:
        print("No proteins found for this taxonomy ID")
proteinDF.to_csv('medTempProteins.csv', index=False)


print('done')