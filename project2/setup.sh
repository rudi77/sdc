# download traffic sign dataset
wget https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip

TFSDPATH="traffic-signs-data"

mkdir $TFSDPATH

mv ./traffic-signs-data.zip $TFSDPATH

unzip $TFSDPATH/traffic-signs-data.zip -d $TFSDPATH

rm $TFSDPATH/traffic-signs-data.zip

mkdir traffic-signs-data-augmented

