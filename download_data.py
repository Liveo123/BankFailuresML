import os
import shutil
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import io
from apiclient.http import MediaIoBaseDownload

# If modifying these scopes, delete the file token.json.
SCOPES = 'https://www.googleapis.com/auth/drive'

files = [{'filename':'predictions', 'fileId' : '1MwQo4frY-EUT2EcWIkXs_jtzxwKF0XeU'},
		 {'filename':'zip_fips', 'fileId' : '1F-dCiaY8t7sa3CaxAvAbTVE4Gm-iEJOL'},
		 {'filename':'combined_data_v2', 'fileId' : '1lqyyd0Duxej1NEQZ9X_Zr3PJT47FVYcR'},
		 {'filename':'banklist', 'fileId' : '15mRJjh9FcVl9do4eG3-vvXd_eZzC3QzG'},
		 {'filename':'combined_data', 'fileId' : '11dFFYLBdedDe2a_CUERATU06KgjhqoBz'},
]

def create_data_dir():
	print("creating directory")
	shutil.rmtree('data', ignore_errors=True)
	os.makedirs('data')

def download_with_google_drive_api(*args, **kwargs):
    store = file.Storage('tokenWrite.json')
    creds = store.get()
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
        creds = tools.run_flow(flow, store)
    service = build('drive', 'v3', http=creds.authorize(Http()))

    # Call the Drive v3 API
    # results = service.files().list(
    #     pageSize=10, fields="nextPageToken, files(id, name)").execute()
    # items = results.get('files', [])

    # if not items:
    #     print('No files found.')
    # else:
    #     print('Files:')
    #     for item in items:
    #         print(u'{0} ({1})'.format(item['name'], item['id']))


    for f in files:
	    # file_id = 
	    file_path = 'data/{filename}.csv'.format(**f)
	    request = service.files().get_media(fileId=f['fileId'])
	    fh = io.FileIO(file_path, mode='wb')
	    downloader = MediaIoBaseDownload(fh, request)
	    done = False
	    while done is False:
	    	status, done = downloader.next_chunk()
	    	print("Download {} {}%.".format(f['filename'], int(status.progress() * 100)))


if __name__ == "__main__":
	create_data_dir()
	download_with_google_drive_api()