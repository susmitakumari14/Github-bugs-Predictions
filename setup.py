import os
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set NLTK data path to the current directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

def download_nltk_data():
    """Download required NLTK data to the specified directory."""
    try:
        print(f"Downloading NLTK data to {nltk_data_dir}...")
        # Download punkt first and verify
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)
        nltk.download('universal_tagset', download_dir=nltk_data_dir, quiet=True)
        
        packages = ['stopwords', 'wordnet', 'omw-1.4']
        for package in packages:
            try:
                nltk.download(package, download_dir=nltk_data_dir, quiet=True)
                print(f"Successfully downloaded {package}")
            except Exception as e:
                print(f"Error downloading {package}: {str(e)}")
        
        # Verify punkt is working
        from nltk.tokenize import word_tokenize
        test_text = "Testing tokenization."
        tokens = word_tokenize(test_text)
        print("NLTK initialization successful!")
        
    except Exception as e:
        print(f"Error during NLTK data download: {str(e)}")
        raise

if __name__ == "__main__":
    download_nltk_data()
else:
    # When imported as a module, still ensure data is downloaded
    download_nltk_data()
