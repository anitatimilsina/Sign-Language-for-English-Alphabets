from zipfile import ZipFile

zf = ZipFile('./data/sign-language-for-alphabets.zip', 'r')
zf.extractall('./data')
zf.close()