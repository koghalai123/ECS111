import pandas as pd
import requests, sys, json

# Load the CSV file
file_path = 'FinalProject/200617_TEMPURA.csv'  # Replace with the actual path to your CSV file
data = pd.read_csv(file_path)

#print(data.head())

#sorted_data = data.sort_values(by='Topt_ave')  # Replace 'column_name' with the actual column name

#print(sorted_data.head())

unique_elements = data['Topt_ave'].unique()  # Replace 'column_name' with the actual column name


lowTempEntries = data[(data['Topt_ave'] <= 10)]

medTempEntries = data[(data['Topt_ave'] >= 20) & (data['Topt_ave'] <= 40)]

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

# Get the data
taxonomy_id = "54252"  # Candidatus Methanomethylophilus alvus
print(f"Querying proteins for taxonomy ID: {taxonomy_id}")
proteins = get_uniprot_proteins(taxonomy_id)

# Convert to DataFrame
if proteins:
    df = pd.DataFrame([flatten_protein_data(p) for p in proteins])
    print(f"\nFound {len(df)} proteins for taxonomy ID {taxonomy_id}")
    print(df.head())
    
    # Verify all results match our taxonomy ID
    mismatches = df[df['Taxonomy ID'].astype(str) != str(taxonomy_id)]
    if not mismatches.empty:
        print(f"\nWarning: Found {len(mismatches)} proteins with incorrect taxonomy IDs:")
        print(mismatches[['Accession', 'Taxonomy ID']])
else:
    print("No proteins found for this taxonomy ID")

# Get the data
taxonomy_id = "54252"  # Example: Candidatus Methanomethylophilus alvus
proteins = get_uniprot_proteins(taxonomy_id)

# Convert to DataFrame
df = pd.DataFrame([flatten_protein_data(p) for p in proteins])

# Display the first few rows
print(df.head())



print('done')