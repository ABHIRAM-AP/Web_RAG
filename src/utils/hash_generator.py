import hashlib

def generate_content_hash(content:str)->str:
    """Hash actual text content"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def generate_doc_id(file_hash:str,index:int)->str:

    return hashlib.sha256(f"{file_hash}_{index}".encode('utf-8')).hexdigest()

