from os import getcwd, path
from zipfile import ZipFile
from io import BytesIO
try:
    from requests import Session
except ModuleNotFoundError:
    print('Please install "requests" module')

current_directory = getcwd()
    
def download_from_gdrive(share_link):
    file_id = get_google_file_id(share_link)
    URL = "https://docs.google.com/uc?export=download"
    session = Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    return response

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def get_google_file_id(url):
    id_start = url.find('open?id=') + len('open?id=')
    file_id = url[id_start:]
    return file_id

def extract_zip(file_raw):
    file_zip = ZipFile(BytesIO(file_raw.content))
    print(str(file_zip.namelist()))
    file_zip.extractall()
    print('Download to notebook directory completed: {}'.format(current_directory))

def kaggle_from_google(gdrive_url, file_list):
    if any(not(path.exists(file)) for file in file_list):
        print('Data File(s) not found')
        print('Downloading...')
        file_raw = download_from_gdrive(gdrive_url)
        extract_zip(file_raw)
    else:
        print('Data Files already downloaded in {}'.format(current_directory))