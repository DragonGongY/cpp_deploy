# download pre-compiled paddle encrypt
ENCRYPTION_URL=https://bj.bcebos.com/paddlex/tools/paddlex-encryption.zip
if [ ! -d "./paddlex-encryption" ]; then
    wget -c ${ENCRYPTION_URL}
    unzip paddlex-encryption.zip
    rm -rf paddlex-encryption.zip
fi
