import os

def load_text_articles(directory_path: str):
    documents = []
    if os.path.exists(directory_path):
        for file_name in os.listdir(directory_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(directory_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({"text": content, "source": file_name})
    else:
        print(f"Directory not found: {directory_path}")
    return documents

def get_all_documents(base_data_dir: str):
    docs = []
    
    wiki_dir = os.path.join(base_data_dir, "medical_articles")
    pubmed_dir = os.path.join(base_data_dir, "pubmed_articles")
    processed_dataset_dir = os.path.join(base_data_dir, "processed_dataset")
    
    print(f"Loading from {wiki_dir}...")
    docs.extend(load_text_articles(wiki_dir))
    
    print(f"Loading from {pubmed_dir}...")
    docs.extend(load_text_articles(pubmed_dir))
    
    print(f"Loading from {processed_dataset_dir}...")
    docs.extend(load_text_articles(processed_dataset_dir))
    
    return docs
