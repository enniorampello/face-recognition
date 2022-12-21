from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()           
drive = GoogleDrive(gauth)  

# upload the file README.md to google drive
file1 = drive.CreateFile({'title': 'README.md'})
file1.SetContentFile('README.md')
file1.Upload()
