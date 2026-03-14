import os
import time
import requests
import wikipedia
import pandas as pd

def fetch_wikipedia_articles(topics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Fetching Wikipedia Articles ---")
    for topic in topics:
        file_name = f"{topic.replace(' ', '_')}.txt"
        file_path = os.path.join(output_dir, file_name)
        if os.path.exists(file_path):
            print(f"Skipping '{topic}', already exists.")
            continue
        try:
            print(f"Fetching: {topic}")
            search_results = wikipedia.search(topic)
            if not search_results:
                print(f"  No Wikipedia results found for {topic}")
                continue
            
            # Fetch the first matched page
            page = wikipedia.page(search_results[0])
            
            content = f"Title: {page.title}\n\nSummary:\n{page.summary}\n\n"
            content += "Main Section Extract:\n"
            content += page.content[:2000] # Grab some extra context from the article body
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"  Error fetching Wikipedia for '{topic}': {e}")
        time.sleep(1) # Delay to respect API limits

def fetch_pubmed_abstracts(topics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Fetching PubMed Abstracts ---")
    
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    for topic in topics:
        file_name = f"{topic.replace(' ', '_')}.txt"
        file_path = os.path.join(output_dir, file_name)
        if os.path.exists(file_path):
            print(f"Skipping '{topic}', already exists.")
            continue
            
        try:
            print(f"Fetching PubMed: {topic}")
            search_params = {
                "db": "pubmed",
                "term": f"{topic}[Title/Abstract]",
                "retmax": 3,
                "retmode": "json"
            }
            search_res = requests.get(search_url, params=search_params)
            search_res.raise_for_status()
            
            id_list = search_res.json().get("esearchresult", {}).get("idlist", [])
            
            if not id_list:
                print(f"  No PubMed results for {topic}")
                continue
                
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "rettype": "abstract",
                "retmode": "text"
            }
            fetch_res = requests.get(fetch_url, params=fetch_params)
            fetch_res.raise_for_status()
            
            abstracts = fetch_res.text
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Topic: {topic}\n\nPUBMED ABSTRACTS:\n\n{abstracts}")
        except Exception as e:
            print(f"  Error fetching PubMed for '{topic}': {e}")
        time.sleep(1)

def process_symptom_dataset(dataset_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Processing Multimodal Symptom Dataset ---")
    
    dataset_csv = os.path.join(dataset_dir, "dataset.csv")
    desc_csv = os.path.join(dataset_dir, "symptom_Description.csv")
    prec_csv = os.path.join(dataset_dir, "symptom_precaution.csv")
    
    if not os.path.exists(dataset_csv):
        print(f"Dataset CSV not found at {dataset_csv}, skipping processing.")
        return
        
    df_sym = pd.read_csv(dataset_csv)
    df_desc = pd.read_csv(desc_csv) if os.path.exists(desc_csv) else pd.DataFrame(columns=["Disease", "Description"])
    df_prec = pd.read_csv(prec_csv) if os.path.exists(prec_csv) else pd.DataFrame(columns=["Disease"])

    # Clean stripping for matching
    df_sym['Disease'] = df_sym['Disease'].str.strip()
    df_desc['Disease'] = df_desc['Disease'].str.strip()
    df_prec['Disease'] = df_prec['Disease'].str.strip()

    # Get unique diseases and their symptoms
    disease_symptoms = {}
    for _, row in df_sym.iterrows():
        disease = row['Disease']
        symptoms = [str(x).strip().replace('_', ' ') for x in row.values[1:] if pd.notna(x) and str(x).strip() != '']
        if disease not in disease_symptoms:
            disease_symptoms[disease] = set()
        disease_symptoms[disease].update(symptoms)

    for disease, symptoms in disease_symptoms.items():
        file_name = f"{disease.replace(' ', '_').replace('/', '_')}.txt"
        file_path = os.path.join(output_dir, file_name)
        
        # Build description
        desc_row = df_desc[df_desc['Disease'] == disease]
        desc = desc_row.iloc[0]['Description'] if not desc_row.empty else "No description available."
        
        # Build precautions
        prec_row = df_prec[df_prec['Disease'] == disease]
        precautions = []
        if not prec_row.empty:
            prec_vals = [str(x).strip() for x in prec_row.iloc[0].values[1:] if pd.notna(x)]
            precautions = prec_vals
            
        content = f"Disease: {disease}\n\nSymptoms: {', '.join(symptoms)}.\n\nDescription: {desc}\n"
        if precautions:
            content += f"\nPrecautions: {', '.join(precautions)}."
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Processed cohesive dataset entry: {disease}")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, "data")
    
    wiki_dir = os.path.join(data_dir, "medical_articles")
    pubmed_dir = os.path.join(data_dir, "pubmed_articles")
    csv_out_dir = os.path.join(data_dir, "processed_dataset")
    dataset_dir = os.path.join(data_dir, "DISEASE_SYMPTOM")
    
    topics = [
        "heat rash", "eczema", "ringworm", "allergic dermatitis",
        "fungal infection", "chickenpox", "measles rash", "viral fever",
        "influenza", "common cold", "skin infection", "contact dermatitis",
        "impetigo", "scabies", "urticaria", "psoriasis", "acne",
        "insect bites", "sunburn", "drug allergy rash"
    ]
    
    process_symptom_dataset(dataset_dir, csv_out_dir)
    fetch_wikipedia_articles(topics, wiki_dir)
    fetch_pubmed_abstracts(topics, pubmed_dir)
    
    print("\nKnowledge ingestion complete. All text files are ready for embedding.")

if __name__ == "__main__":
    main()
