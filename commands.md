## Start Docker compose
```shell
cd docker
docker-compose up

# open Open Search Dashboard
http://localhost:5601

# completely remove previous containers
docker-compose rm
```


## Download data from Kaggle
```shell
# Set up Kaggle API
touch /Users/vitalii.mishchenko/.kaggle/kaggle.json
chmod 600 /Users/vitalii.mishchenko/.kaggle/kaggle.json

# Create folder that contains datasets 
mkdir -p /Users/vitalii.mishchenko/Documents/personal/opensearch/data
cd /Users/vitalii.mishchenko/Documents/personal/opensearch/data

# Download data
kaggle competitions download -c acm-sf-chapter-hackathon-big
unzip acm-sf-chapter-hackathon-big.zip
tar -xf product_data.tar.gz
# Cleaning up to save space
rm acm-sf-chapter-hackathon-big.zip
rm product_data.tar.gz
rm popular_skus.csv
```


## Index documents
```shell
# activate environment
source venv/bin/activate

# index documents
./index-data.sh

# deactivate environment
deactivate
```


## Check Indexing logs 
```shell
# check products logs
tail -f /Users/vitalii.mishchenko/Documents/personal/opensearch/data/logs/index_products.log

# check queries logs
tail -f /Users/vitalii.mishchenko/Documents/personal/opensearch/data/logs/index_queries.log

# check annotations logs
tail -f /Users/vitalii.mishchenko/Documents/personal/opensearch/data/logs/index_annotations.log
```


## Sync with upstream
```shell
# add upstream
git remote add upstream https://github.com/gsingers/search_with_machine_learning_course.git

# pull
git remote pull upstream
```